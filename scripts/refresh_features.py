"""Run configured feature SQL groups for one month."""

from configs.configs_val.script_config import load_script_settings
from src.resources.update_features import FEATURE_SQL_FILES, run_feature_etl


def main() -> None:
    """Run the feature groups selected in ``configs/scripts.yaml``."""
    config = load_script_settings().refresh_features
    unknown_groups = set(config.groups or ()) - set(FEATURE_SQL_FILES)
    if unknown_groups:
        raise ValueError(f"Unknown feature groups in configs/scripts.yaml: {sorted(unknown_groups)}")

    results = run_feature_etl(
        config.ym,
        dry_run=config.dry_run,
        groups=list(config.groups) if config.groups else None,
        progress=lambda message: print(message, flush=True),
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
