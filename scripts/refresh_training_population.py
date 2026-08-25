"""Rebuild configured training-population snapshots from one latest month."""

from configs.configs_val.product_config import PRODUCT_CONFIGS
from configs.configs_val.script_config import load_script_settings
from src.resources.sampling import ensure_training_populations


def main() -> None:
    config = load_script_settings().refresh_training_population
    products = config.product
    unknown_products = set(products or ()) - set(PRODUCT_CONFIGS)
    if unknown_products:
        raise ValueError(f"Unknown products in configs/scripts.yaml: {sorted(unknown_products)}")
    product_configs = [PRODUCT_CONFIGS[name] for name in products] if products else None
    result = ensure_training_populations(config.ym, configs=product_configs)
    print(
        f"prediction_ym={result.prediction_ym} "
        f"inserted_rows={result.inserted_count:,} "
        f"existing_rows={result.existing_count:,} "
        f"inserted_snapshots={result.inserted_snapshots:,} "
        f"reused_snapshots={result.reused_snapshots:,}"
    )


if __name__ == "__main__":
    main()
