"""Presentation-ready, aggregate-only model performance reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.evaluation import EvaluationResult


def performance_summary(
    train: EvaluationResult,
    backtest: EvaluationResult,
) -> dict[str, Any]:
    """Build the common metrics stored in the model training summary."""
    return {
        "train_auc": train.auc,
        "backtest_auc": backtest.auc,
        "auc_gap": train.auc - backtest.auc,
        "train_ks": train.ks,
        "backtest_ks": backtest.ks,
        "train_sample": {
            "count": train.row_count,
            "positive_count": train.positive_count,
            "base_rate": train.base_rate,
        },
        "backtest_sample": {
            "count": backtest.row_count,
            "positive_count": backtest.positive_count,
            "base_rate": backtest.base_rate,
        },
        "backtest_top_rates": backtest.top_rate_metrics,
        "backtest_deciles": backtest.decile_metrics,
        "backtest_levels": backtest.level_metrics,
    }


def _percent(value: float | int) -> str:
    return f"{float(value):.1%}"


def _model_card(
    summary: dict[str, Any],
    importance: pd.DataFrame,
) -> str:
    metrics = summary["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("summary.metrics must be a mapping")
    train_sample = metrics["train_sample"]
    backtest_sample = metrics["backtest_sample"]
    top_rates = metrics["backtest_top_rates"]
    deciles = metrics["backtest_deciles"]
    if not isinstance(train_sample, dict) or not isinstance(backtest_sample, dict):
        raise TypeError("sample metrics must be mappings")
    if not isinstance(top_rates, list) or not isinstance(deciles, list):
        raise TypeError("business metrics must be lists")

    lines = [
        f"# {summary['product']}模型成效摘要",
        "",
        f"- 客群：{summary['population']}",
        f"- 模型版本：{summary['edition']}",
        f"- 訓練月份：{', '.join(summary['periods']['train_yms'])}",
        f"- 回測月份：{summary['periods']['backtest_ym']}",
        f"- 驗收結果：{'通過' if summary['passed'] else '未通過'}",
        "",
        "## 整體表現",
        "",
        "| 指標 | 訓練集 | 回測集 |",
        "|---|---:|---:|",
        f"| 樣本數 | {int(train_sample['count']):,} | {int(backtest_sample['count']):,} |",
        f"| 流失人數 | {int(train_sample['positive_count']):,} | {int(backtest_sample['positive_count']):,} |",
        f"| 流失率 | {_percent(train_sample['base_rate'])} | {_percent(backtest_sample['base_rate'])} |",
        f"| AUC | {float(metrics['train_auc']):.4f} | {float(metrics['backtest_auc']):.4f} |",
        f"| KS | {float(metrics['train_ks']):.4f} | {float(metrics['backtest_ks']):.4f} |",
        "",
        f"訓練與回測 AUC 差距：{float(metrics['auc_gap']):.4f}",
        "",
        "## 優先聯繫客群成效",
        "",
        "| 優先比例 | 人數 | 流失人數 | 命中率 | Recall | Lift |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_rates:
        lines.append(
            f"| Top {float(row['top_percent']):g}% | {int(row['selected_count']):,} | "
            f"{int(row['positive_count']):,} | {_percent(row['hit_rate'])} | "
            f"{_percent(row['recall'])} | {float(row['lift']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## 十等分成效",
            "",
            "第 1 等分代表模型分數最高、最優先關懷的 10% 客群。",
            "",
            "| 等分 | 人數 | 流失人數 | 命中率 | Lift | 累積 Recall |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deciles:
        lines.append(
            f"| {int(row['decile'])} | {int(row['count']):,} | {int(row['positive_count']):,} | "
            f"{_percent(row['hit_rate'])} | {float(row['lift']):.2f} | "
            f"{_percent(row['cumulative_recall'])} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 重要特徵",
            "",
            "| 排名 | 特徵 | Importance |",
            "|---:|---|---:|",
        ]
    )
    rows = importance.loc[:, ["feature", "importance"]].head(10).itertuples(index=False, name=None)
    for rank, (feature, importance_value) in enumerate(rows, start=1):
        lines.append(f"| {rank} | {feature} | {float(importance_value):.4f} |")

    if summary.get("development_sample", {}).get("limited"):
        rows = summary["development_sample"]["rows_per_month"]
        lines.extend(["", f"> 注意：此為流程測試模型，每月限制 {int(rows):,} 筆，不代表正式全量結果。"])
    lines.extend(["", "> 目前只有一個 OOT 回測月份，跨月份穩定性仍需增加其他回測月確認。", ""])
    return "\n".join(lines)


def write_performance_report(
    artifact_dir: Path,
    summary: dict[str, Any],
    importance: pd.DataFrame,
) -> None:
    """Write presentation-ready aggregate tables and a Chinese model card."""
    metrics = summary["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("summary.metrics must be a mapping")
    pd.DataFrame(metrics["backtest_top_rates"]).to_csv(
        artifact_dir / "business_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metrics["backtest_deciles"]).to_csv(
        artifact_dir / "score_deciles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (artifact_dir / "model_card.md").write_text(_model_card(summary, importance), encoding="utf-8")
