"""Refresh one configured full-population month independently."""

from configs.configs_val.product_config import get_product_config
from configs.configs_val.script_config import load_script_settings
from src.resources.population import population_snapshot_ym, refresh_population


def main() -> None:
    config = load_script_settings().refresh_population
    product_config = get_product_config(config.product)
    snapshot_ym = population_snapshot_ym(product_config, config.as_of_ym)
    print(
        f"Derived snapshot: as_of_ym={config.as_of_ym}, product={config.product}, "
        f"label_offset_months={product_config.training_schedule.label_offset_months}, snapshot_ym={snapshot_ym}"
    )

    print(refresh_population(snapshot_ym, replace=True))


if __name__ == "__main__":
    main()

