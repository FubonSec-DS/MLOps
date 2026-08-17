"""Leakage-safe staged retraining pipeline."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import BaseModel
from xgboost import XGBClassifier

from configs.configs_val.DB_configs import get_settings
from configs.configs_val.product_config import ProductConfig, get_product_config
from src.models.evaluation import check_retrain_passed, evaluate
from src.models.preprocessing import PreprocessingSchema, apply_preprocessing, fit_preprocessing
from src.models.training import fit_xgboost, save_model, select_top_features
from src.models.versioning import save_retrain_outputs
from src.resources.feature_loader import (
    LoadedDataset,
    combine_months,
    discover_feature_catalog,
    load_pretrain_month,
    load_training_month,
)


@dataclass(frozen=True)
class RetrainResult:
    """High-level result returned to CLI and schedulers."""

    product: str  # 產品名稱
    population: str  # 母體類型
    edition: str  # 版本號
    passed: bool  # 模型是否通過驗收
    train_auc: float  # 訓練集 AUC
    backtest_auc: float  # OOT 回測集 AUC
    artifact_dir: str  # 模型存放路徑


@dataclass(frozen=True)
class PretrainReadinessResult:
    """Summary of data immediately before the pretrain XGBoost call."""

    product: str
    population: str
    yyyymm: str
    cohort_rows: int
    joined_columns: int
    candidate_features: int
    categorical_features: int
    y0_count: int
    y1_count: int
    duplicate_keys: int
    target_classes: tuple[int, ...]
    ready: bool
    issues: tuple[str, ...]
    all_null_feature_count: int
    all_null_features: tuple[str, ...]
    matrix_rows: int
    matrix_columns: int
    matrix_memory_mb: float
    report_dir: str | None


@dataclass(frozen=True)
class PretrainResult:
    """Result of ephemeral XGBoost feature selection without formal training."""

    product: str
    population: str
    train_yms: tuple[str, ...]
    rows: int
    candidate_features: int
    selected_features: int
    output_dir: str


def _jsonable(value: object) -> object:
    """Convert runtime configuration objects into JSON-safe values."""
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _object_snapshot(name: str, value: object) -> dict[str, object]:
    object_type = type(value)
    return {
        "name": name,
        "python_type": f"{object_type.__module__}.{object_type.__qualname__}",
        "value": _jsonable(value),
    }


def _column_profile(frame: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Describe every joined column and show a few real, non-null values."""
    rows: list[dict[str, object]] = []
    for name in frame.columns:
        series = cast("pd.Series", frame[name])
        examples = [_jsonable(value) for value in series.dropna().drop_duplicates().head(3).tolist()]
        if name in {"customer_id", "yyyymm"}:
            role = "key"
        elif name == "y":
            role = "target"
        else:
            role = "feature"
        rows.append(
            {
                "column": name,
                "role": role,
                "joined_dtype": str(series.dtype),
                "matrix_dtype": str(matrix[name].dtype) if name in matrix.columns else "",
                "null_count": int(series.isna().sum()),
                "null_percent": round(float(series.isna().mean()) * 100, 4),
                "unique_count": int(series.nunique(dropna=True)),
                "example_values": json.dumps(examples, ensure_ascii=False, default=str),
            }
        )
    return pd.DataFrame(rows)


def _write_pretrain_inspection_report(
    report_dir: Path,
    *,
    result: PretrainReadinessResult,
    config: ProductConfig,
    frame: pd.DataFrame,
    candidates: list[str],
    matrix: pd.DataFrame,
    schema: PreprocessingSchema,
    sample_rows: int,
) -> None:
    """Write human-readable snapshots of the values passed toward pretraining."""
    report_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    population_config = config.get_population(result.population)
    safe_settings = {
        "oracle": {
            "feature_schema": settings.oracle.feature_schema,
            "population_sampling_schema": settings.oracle.population_sampling_schema,
            "population_sampling_table": settings.oracle.population_sampling_table,
            "population_source_schema": settings.oracle.population_source_schema,
            "model_log_schema": settings.oracle.model_log_schema,
            "credentials": "omitted",
        },
        "paths": settings.paths,
        "models": settings.models,
        "configured_categorical_feature_count": len(settings.features.categorical_features),
    }
    runtime_objects = [
        _object_snapshot("product_config", config),
        _object_snapshot("population_config", population_config),
        _object_snapshot("pretrain_params", config.pretrain_params),
        _object_snapshot("preprocessing_schema", schema),
        _object_snapshot("settings_without_credentials", safe_settings),
        _object_snapshot("candidate_features", candidates),
    ]

    _write_json(report_dir / "summary.json", _jsonable(result))
    _write_json(report_dir / "runtime_objects.json", runtime_objects)
    _write_json(report_dir / "preprocessing_schema.json", schema.to_dict())
    _column_profile(frame, matrix).to_csv(report_dir / "columns.csv", index=False, encoding="utf-8-sig")
    frame.head(sample_rows).to_csv(report_dir / "joined_frame_sample.csv", index=False, encoding="utf-8-sig")
    matrix.head(sample_rows).to_csv(report_dir / "preprocessed_matrix_sample.csv", index=False, encoding="utf-8-sig")
    (report_dir / "README.md").write_text(
        """# Pretrain inspection report

這份目錄記錄資料在送進 pretrain XGBoost 前的實際樣子。

- `summary.json`：筆數、正負樣本數、特徵數、矩陣大小與檢查結果。
- `runtime_objects.json`：class 的完整 Python 型別名稱及當次實際設定值；資料庫密碼不輸出。
- `columns.csv`：每個欄位的角色、前後 dtype、NULL、唯一值數及三個實際值。
- `joined_frame_sample.csv`：特徵 join 後、前處理前的實際資料 sample，包含 customer_id。
- `preprocessed_matrix_sample.csv`：真正交給 XGBoost 的 X 矩陣 sample，不含 key 與 y。
- `preprocessing_schema.json`：類別欄位及本次學到的 category vocabulary。

注意：兩個 sample 檔含真實內部資料，只應留在受控環境，不應提交到 Git。
""",
        encoding="utf-8",
    )


def _validate_periods(train_yms: list[str], backtest_ym: str) -> None:
    periods = [*train_yms, backtest_ym]
    if not train_yms:
        raise ValueError("At least one training month is required")
    for ym in periods:
        if len(ym) != 6 or not ym.isdigit() or not 1 <= int(ym[4:]) <= 12:
            raise ValueError(f"Invalid YYYYMM value: {ym!r}")
    if len(set(periods)) != len(periods):
        raise ValueError("Model-building and OOT backtest months must not overlap")
    if periods != sorted(periods):
        raise ValueError("Months must be in chronological order")


def _candidate_features(dataset: LoadedDataset, config: ProductConfig) -> list[str]:
    features = [name for name in dataset.frame.columns if name not in {"customer_id", "yyyymm", "y"}]
    if config.drop_keywords:
        features = [
            name for name in features if not any(keyword.lower() in name.lower() for keyword in config.drop_keywords)
        ]
    if not features:
        raise ValueError("No candidate features remain after filtering")
    return features


def _month_summary(dataset: LoadedDataset) -> dict[str, int | float | str]:
    frame = dataset.frame
    ym = str(frame["yyyymm"].iloc[0]) if not frame.empty else ""
    return {
        "yyyymm": ym,
        "actual_count": len(frame),
        "y1_count": int((frame["y"] == 1).sum()),
        "y0_count": int((frame["y"] == 0).sum()),
    }


def validate_pretrain_readiness(
    *,
    product: str,
    population: str,
    ym: str,
    sample_size: int = 1_000,
    report_dir: Path | None = None,
    sample_rows: int = 5,
) -> PretrainReadinessResult:
    """Build a sampled pretrain matrix without fitting XGBoost."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if sample_rows <= 0:
        raise ValueError("sample_rows must be greater than zero")

    config = get_product_config(product)
    catalog = discover_feature_catalog()
    dataset = load_pretrain_month(
        config,
        population,
        ym,
        catalog=catalog,
        row_limit=sample_size,
    )
    frame = dataset.frame
    if frame.empty:
        raise ValueError(f"No pretrain rows were loaded for {product}/{population}/{ym}")

    duplicate_keys = int(frame.duplicated(["customer_id", "yyyymm"]).sum())
    if duplicate_keys:
        raise ValueError(f"Pretrain join produced {duplicate_keys:,} duplicate customer/month keys")

    target_classes = tuple(sorted(set(frame["y"].dropna().astype(int))))
    invalid_targets = sorted(set(target_classes) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Pretrain target contains values other than 0/1: {invalid_targets}")

    candidates = _candidate_features(dataset, config)
    all_null = tuple(name for name in candidates if cast("bool", frame[name].isna().all()))
    matrix, schema = fit_preprocessing(frame, candidates, dataset.categorical_features)
    if matrix.shape != (len(frame), len(candidates)):
        raise ValueError(f"Unexpected pretrain matrix shape {matrix.shape}; expected {(len(frame), len(candidates))}")

    issues = []
    if set(target_classes) != {0, 1}:
        issues.append(f"Pretrain cohort must contain y=0 and y=1; found {list(target_classes)}")

    result = PretrainReadinessResult(
        product=product,
        population=population,
        yyyymm=ym,
        cohort_rows=len(frame),
        joined_columns=len(frame.columns),
        candidate_features=len(candidates),
        categorical_features=len(schema.categorical_features),
        y0_count=int((frame["y"] == 0).sum()),
        y1_count=int((frame["y"] == 1).sum()),
        duplicate_keys=duplicate_keys,
        target_classes=target_classes,
        ready=not issues,
        issues=tuple(issues),
        all_null_feature_count=len(all_null),
        all_null_features=all_null,
        matrix_rows=matrix.shape[0],
        matrix_columns=matrix.shape[1],
        matrix_memory_mb=round(float(matrix.memory_usage(deep=True).sum()) / 1024**2, 2),
        report_dir=str(report_dir.resolve()) if report_dir is not None else None,
    )
    if report_dir is not None:
        _write_pretrain_inspection_report(
            report_dir,
            result=result,
            config=config,
            frame=frame,
            candidates=candidates,
            matrix=matrix,
            schema=schema,
            sample_rows=sample_rows,
        )
    return result


def _require_binary_target(target: pd.Series, context: str) -> None:
    """Reject model input that does not contain both binary target classes."""
    classes = sorted(set(target.dropna().astype(int)))
    if classes != [0, 1]:
        raise ValueError(f"{context} must contain y=0 and y=1 before XGBoost; found {classes}")


def run_pretrain(
    *,
    product: str,
    population: str,
    train_yms: list[str],
    output_dir: Path,
    rows_per_month: int | None = None,
) -> PretrainResult:
    """Fit only the ephemeral feature-selection model and persist its outputs."""
    if not train_yms:
        raise ValueError("At least one pretrain month is required")
    if rows_per_month is not None and rows_per_month <= 0:
        raise ValueError("rows_per_month must be greater than zero")
    for ym in train_yms:
        if len(ym) != 6 or not ym.isdigit() or not 1 <= int(ym[4:]) <= 12:
            raise ValueError(f"Invalid YYYYMM value: {ym!r}")

    config = get_product_config(product)
    settings = get_settings()
    catalog = discover_feature_catalog()
    months = []
    for index, ym in enumerate(train_yms, start=1):
        limit_text = f"{rows_per_month:,}" if rows_per_month is not None else "production"
        print(f"[{index}/{len(train_yms)}] loading pretrain month {ym} limit={limit_text}", flush=True)
        month = load_pretrain_month(
            config,
            population,
            ym,
            catalog=catalog,
            row_limit=rows_per_month,
        )
        print(f"[{index}/{len(train_yms)}] loaded {len(month.frame):,} rows", flush=True)
        months.append(month)
    data = combine_months(months)
    candidates = _candidate_features(data, config)
    matrix, preprocessing = fit_preprocessing(data.frame, candidates, data.categorical_features)
    target = cast("pd.Series", data.frame["y"]).astype(int)
    _require_binary_target(target, "Pretrain cohort")
    params = config.pretrain_params
    if params.early_stopping_rounds is not None:
        raise ValueError("Pretrain params cannot use early stopping without a separate pretrain validation period")

    print(
        f"fitting pretrain XGBoost rows={len(matrix):,} features={len(candidates):,} y1={int((target == 1).sum()):,}",
        flush=True,
    )
    model = fit_xgboost(
        matrix,
        target,
        params,
        device=settings.models.device,
        random_state=settings.models.random_state,
    )
    selected, importance = select_top_features(model, candidates, top_n=settings.models.top_n_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "selected_features.json", selected)
    importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    _write_json(output_dir / "preprocessing_schema.json", preprocessing.to_dict())
    _write_json(
        output_dir / "pretrain_summary.json",
        {
            "product": product,
            "population": population,
            "train_yms": train_yms,
            "development_rows_per_month": rows_per_month,
            "development_limited": rows_per_month is not None,
            "months": [_month_summary(item) for item in months],
            "rows": len(data.frame),
            "candidate_features": len(candidates),
            "categorical_features": len(preprocessing.categorical_features),
            "selected_features": len(selected),
            "target": {"y0": int((target == 0).sum()), "y1": int((target == 1).sum())},
            "matrix_memory_mb": round(float(matrix.memory_usage(deep=True).sum()) / 1024**2, 2),
            "params": params.model_dump(),
            "device": settings.models.device,
            "random_state": settings.models.random_state,
            "model_persisted": False,
        },
    )
    return PretrainResult(
        product=product,
        population=population,
        train_yms=tuple(train_yms),
        rows=len(data.frame),
        candidate_features=len(candidates),
        selected_features=len(selected),
        output_dir=str(output_dir.resolve()),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_artifacts(
    artifact_dir: Path,
    *,
    model: XGBClassifier,
    selected_features: list[str],
    importance: pd.DataFrame,
    preprocessing: PreprocessingSchema,
    summary: dict,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, artifact_dir / "model.pickle")
    _write_json(artifact_dir / "features.json", selected_features)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    _write_json(artifact_dir / "preprocessing.json", preprocessing.to_dict())
    _write_json(artifact_dir / "training_summary.json", summary)


def run_retrain(
    *,
    product: str,
    population: str,
    train_yms: list[str],
    backtest_ym: str,
    edition: str,
    write_db: bool = False,
) -> RetrainResult:
    """Run feature selection, formal training, and one untouched OOT backtest."""
    _validate_periods(train_yms, backtest_ym)
    config = get_product_config(product)
    settings = get_settings()
    artifact_dir = settings.paths.model_base / product / population / edition
    if artifact_dir.exists():
        raise FileExistsError(f"Model artifact directory already exists: {artifact_dir}")

    # 探索可用的特徵欄位
    catalog = discover_feature_catalog()

    # ----------- pretrain ----------------
    pretrain_months = [load_pretrain_month(config, population, ym, catalog=catalog) for ym in train_yms]
    pretrain_data = combine_months(pretrain_months)

    for dataset in pretrain_months:
        if len(dataset.frame) > settings.models.pretrain_size:
            ym = str(dataset.frame["yyyymm"].iloc[0])
            warnings.warn(
                f"{ym} pretrain cohort exceeds {settings.models.pretrain_size:,} because all y=1 rows were kept",
                stacklevel=2,
            )
    candidates = _candidate_features(pretrain_data, config)
    if len(candidates) < settings.models.top_n_features:
        warnings.warn(
            f"Only {len(candidates)} candidate features are available; all will be used",
            stacklevel=2,
        )
    X_pretrain, _ = fit_preprocessing(
        pretrain_data.frame,
        candidates,
        pretrain_data.categorical_features,
    )
    y_pretrain = cast("pd.Series", pretrain_data.frame["y"]).astype(int)
    _require_binary_target(y_pretrain, "Pretrain cohort")
    pretrain_params = config.pretrain_params
    if pretrain_params.early_stopping_rounds is not None:
        raise ValueError("Pretrain params cannot use early stopping without a separate pretrain validation period")
    pretrain_model = fit_xgboost(
        X_pretrain,
        y_pretrain,
        pretrain_params,
        device=settings.models.device,
        random_state=settings.models.random_state,
    )
    selected_features, importance = select_top_features(
        pretrain_model,
        candidates,
        top_n=settings.models.top_n_features,
    )

    # ----------- 訓練 ----------------
    train_months = [load_training_month(config, population, ym, selected_features, catalog=catalog) for ym in train_yms]
    backtest_data = load_training_month(
        config,
        population,
        backtest_ym,
        selected_features,
        catalog=catalog,
    )
    train_data = combine_months(train_months)
    X_train, preprocessing = fit_preprocessing(
        train_data.frame,
        selected_features,
        train_data.categorical_features,
    )
    X_backtest = apply_preprocessing(backtest_data.frame, selected_features, preprocessing)
    y_train = cast("pd.Series", train_data.frame["y"]).astype(int)
    y_backtest = cast("pd.Series", backtest_data.frame["y"]).astype(int)
    _require_binary_target(y_train, "Training cohort")
    _require_binary_target(y_backtest, "OOT backtest cohort")

    training_params = config.training_params
    if training_params.early_stopping_rounds is not None:
        raise ValueError("Two-stage training params cannot use early stopping without a validation set")
    model = fit_xgboost(
        X_train,
        y_train,
        training_params,
        device=settings.models.device,
        random_state=settings.models.random_state,
    )
    train_evaluation = evaluate(model, X_train, y_train, config.bins, config.rank_bins)
    backtest_evaluation = evaluate(model, X_backtest, y_backtest, config.bins, config.rank_bins)
    passed = check_retrain_passed(backtest_evaluation.auc)

    summary = {
        "product": product,
        "business_target": config.business_target,
        "population": population,
        "training_population_source": {
            "schema": settings.oracle.population_sampling_schema,
            "table": settings.oracle.population_sampling_table,
        },
        "edition": edition,
        "created_at": datetime.now().astimezone().isoformat(),
        "periods": {
            "train_yms": train_yms,
            "backtest_ym": backtest_ym,
        },
        "feature_count": len(selected_features),
        "pretrain_months": [_month_summary(item) for item in pretrain_months],
        "training_months": [_month_summary(item) for item in train_months],
        "backtest": _month_summary(backtest_data),
        "params": {
            "pretrain": pretrain_params.model_dump(),
            "training": training_params.model_dump(),
            "device": settings.models.device,
            "random_state": settings.models.random_state,
        },
        "metrics": {
            "train_auc": train_evaluation.auc,
            "backtest_auc": backtest_evaluation.auc,
            "backtest_levels": backtest_evaluation.level_metrics,
        },
        "passed": passed,
    }
    _write_artifacts(
        artifact_dir,
        model=model,
        selected_features=selected_features,
        importance=importance,
        preprocessing=preprocessing,
        summary=summary,
    )
    if write_db:
        retrain_log = pd.DataFrame(
            [
                {
                    "模型名稱": f"{product}模型",
                    "母體": population,
                    "retrain日期": summary["created_at"],
                    "train_period": str(train_yms),
                    "test_period": backtest_ym,
                    "train_auc": train_evaluation.auc,
                    "test_auc": backtest_evaluation.auc,
                    "訓練母體人數": len(train_data.frame),
                    "訓練母體y=1人數": int((y_train == 1).sum()),
                    "TEST_人數": len(backtest_data.frame),
                    "是否通過標準": "Y" if passed else "N",
                    "版本": edition,
                }
            ]
        )
        model_log = pd.DataFrame(
            [
                {
                    "模型名稱": f"{product}模型",
                    "母體": population,
                    "target": config.business_target,
                    "algorithm": "xgboost",
                    "bins": str(list(config.bins)),
                    "版本": edition,
                    "retrain日期": summary["created_at"],
                }
            ]
        )
        save_retrain_outputs(retrain_log, model_log)

    return RetrainResult(
        product=product,
        population=population,
        edition=edition,
        passed=passed,
        train_auc=train_evaluation.auc,
        backtest_auc=backtest_evaluation.auc,
        artifact_dir=str(artifact_dir),
    )
