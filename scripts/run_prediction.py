"""Score one product/population/month artifact configured in YAML."""

from configs.configs_val.script_config import load_script_settings
from src.pipelines.predict import run_prediction


def main() -> None:
    config = load_script_settings().run_prediction
    result = run_prediction(
        product=config.product,
        population=config.population,
        ym=config.ym,
        edition=config.edition,
        write_db=config.write_db,
    )
    print(result.head())
    print(f"Scored {len(result):,} customers")


if __name__ == "__main__":
    main()

