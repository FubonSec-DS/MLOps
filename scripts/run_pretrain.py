"""Run configured ephemeral XGBoost feature selection without formal training."""

from datetime import datetime

from configs.configs_val.product_config import calculate_training_periods, get_product_config
from configs.configs_val.script_config import load_script_settings, resolve_project_path
from src.pipelines.retrain import run_pretrain


def main() -> None:
    config = load_script_settings().run_pretrain

    product_config = get_product_config(config.product)
    train_yms, backtest_ym, _latest = calculate_training_periods(
        product_config,
        datetime.strptime(config.as_of_ym, "%Y%m"),
    )
    output_dir = (
        resolve_project_path(config.output_dir)
        / config.product
        / config.population
        / config.as_of_ym
        / "pretrain"
    )
    print(f"train_yms={train_yms} excluded_oot_backtest_ym={backtest_ym}", flush=True)
    print(
        run_pretrain(
            product=config.product,
            population=config.population,
            train_yms=train_yms,
            output_dir=output_dir,
            rows_per_month=config.rows_per_month,
        )
    )


if __name__ == "__main__":
    main()

