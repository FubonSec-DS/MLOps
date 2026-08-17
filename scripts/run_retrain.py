"""Run configured two-stage temporal retraining jobs."""

from datetime import datetime

from configs.configs_val.product_config import PRODUCT_CONFIGS, ProductConfig, calculate_training_periods
from configs.configs_val.script_config import RunRetrainConfig, load_script_settings
from src.pipelines.retrain import run_retrain


def _names(selection: str | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Normalize a scalar or list selection while preserving null as 'all'."""
    if selection is None:
        return None
    return (selection,) if isinstance(selection, str) else selection


def _build_jobs(config: RunRetrainConfig) -> list[tuple[ProductConfig, str]]:
    """Expand configured product and population selectors into concrete jobs."""
    selected_products = _names(config.product)
    product_names = selected_products or tuple(PRODUCT_CONFIGS)
    unknown_products = set(product_names) - set(PRODUCT_CONFIGS)
    if unknown_products:
        raise ValueError(f"Unknown retrain products in configs/scripts.yaml: {sorted(unknown_products)}")

    selected_populations = _names(config.population)
    matched_populations: set[str] = set()
    jobs: list[tuple[ProductConfig, str]] = []
    for product_name in product_names:
        product_config = PRODUCT_CONFIGS[product_name]
        configured_populations = tuple(item.population for item in product_config.populations)
        populations = (
            configured_populations
            if selected_populations is None
            else tuple(name for name in selected_populations if name in configured_populations)
        )
        matched_populations.update(populations)
        jobs.extend((product_config, population) for population in populations)

    unmatched_populations = set(selected_populations or ()) - matched_populations
    if unmatched_populations:
        raise ValueError(
            f"Retrain populations are not configured for any selected product: {sorted(unmatched_populations)}"
        )
    if not jobs:
        raise ValueError("No product/population retrain jobs were selected")
    return jobs


def _periods(config: RunRetrainConfig, product_config: ProductConfig) -> tuple[list[str], str, str | None]:
    """Return explicit or product-specific automatically derived periods."""
    if config.train_yms is not None and config.backtest_ym is not None:
        return list(config.train_yms), config.backtest_ym, config.as_of_ym

    reference_date = datetime.strptime(config.as_of_ym, "%Y%m").date() if config.as_of_ym else None
    auto_train_yms, auto_backtest_ym, latest_ym = calculate_training_periods(product_config, reference_date)
    train_yms = list(config.train_yms) if config.train_yms else auto_train_yms
    return train_yms, config.backtest_ym or auto_backtest_ym, latest_ym


def main() -> None:
    config = load_script_settings().run_retrain
    jobs = _build_jobs(config)
    results = []

    for index, (product_config, population) in enumerate(jobs, start=1):
        train_yms, backtest_ym, as_of_ym = _periods(config, product_config)
        print(
            f"[{index}/{len(jobs)}] retraining product={product_config.product} "
            f"population={population} edition={config.edition}",
            flush=True,
        )
        if config.train_yms is None or config.backtest_ym is None:
            print(
                f"  as_of_ym={as_of_ym} train_yms={train_yms} backtest_ym={backtest_ym}",
                flush=True,
            )
        result = run_retrain(
            product=product_config.product,
            population=population,
            train_yms=train_yms,
            backtest_ym=backtest_ym,
            edition=config.edition,
            write_db=config.write_db,
        )
        results.append(result)
        print(result, flush=True)

    passed = sum(result.passed for result in results)
    print(f"Retrain batch complete: jobs={len(results)} passed={passed} failed_acceptance={len(results) - passed}")


if __name__ == "__main__":
    main()
