"""Validate configured joined and preprocessed data immediately before XGBoost."""

import json
from dataclasses import asdict

from configs.configs_val.script_config import load_script_settings, resolve_project_path
from src.pipelines.retrain import validate_pretrain_readiness


def main() -> None:
    """Run one bounded pretrain readiness check."""
    config = load_script_settings().validate_pretrain_readiness
    report_dir = resolve_project_path(config.output_dir) / config.product / config.population / config.ym

    result = validate_pretrain_readiness(
        product=config.product,
        population=config.population,
        ym=config.ym,
        sample_size=config.sample_size,
        report_dir=report_dir,
        sample_rows=config.sample_rows,
    )
    output = asdict(result)
    all_null = output["all_null_features"]
    output["all_null_features"] = all_null[:20]
    output["all_null_features_truncated"] = len(all_null) > 20
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nInspection files: {result.report_dir}")
    if not result.ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

