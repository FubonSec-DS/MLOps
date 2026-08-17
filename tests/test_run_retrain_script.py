"""Tests for YAML-driven single and batch retraining orchestration."""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from typing import cast
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from configs.configs_val.product_config import ProductConfig
from configs.configs_val.script_config import RunRetrainConfig
from scripts.run_retrain import _build_jobs, main


def _product(name: str, *populations: str) -> ProductConfig:
    return cast(
        "ProductConfig",
        SimpleNamespace(
            product=name,
            populations=tuple(SimpleNamespace(population=value) for value in populations),
        ),
    )


def _config(
    *,
    product: str | tuple[str, ...] | None = None,
    population: str | tuple[str, ...] | None = None,
) -> RunRetrainConfig:
    return RunRetrainConfig(
        product=product,
        population=population,
        train_yms=None,
        backtest_ym=None,
        as_of_ym="202607",
        edition="v1",
        write_db=False,
    )


class RetrainScriptTests(TestCase):
    """Protect batch expansion and product-specific schedule derivation."""

    def test_null_selectors_expand_every_product_and_its_populations(self) -> None:
        products = {"A": _product("A", "P1", "P2"), "B": _product("B", "P3")}
        with patch("scripts.run_retrain.PRODUCT_CONFIGS", products):
            jobs = _build_jobs(_config())

        self.assertEqual(
            [(item.product, population) for item, population in jobs],
            [("A", "P1"), ("A", "P2"), ("B", "P3")],
        )

    def test_explicit_population_filters_to_supporting_products(self) -> None:
        products = {"A": _product("A", "P1", "P2"), "B": _product("B", "P2", "P3")}
        with patch("scripts.run_retrain.PRODUCT_CONFIGS", products):
            jobs = _build_jobs(_config(population="P2"))

        self.assertEqual([(item.product, population) for item, population in jobs], [("A", "P2"), ("B", "P2")])

    @patch("scripts.run_retrain.run_retrain")
    @patch("scripts.run_retrain.calculate_training_periods")
    @patch("scripts.run_retrain.load_script_settings")
    def test_main_derives_each_products_schedule_and_runs_every_job(
        self,
        load_settings: MagicMock,
        calculate_periods: MagicMock,
        retrain: MagicMock,
    ) -> None:
        products = {"A": _product("A", "P1", "P2"), "B": _product("B", "P3")}
        load_settings.return_value = SimpleNamespace(run_retrain=_config())
        calculate_periods.side_effect = lambda product, _date: ([f"{product.product}01"], "202506", "202607")
        retrain.side_effect = lambda **values: SimpleNamespace(passed=True, **values)

        with patch("scripts.run_retrain.PRODUCT_CONFIGS", products), redirect_stdout(StringIO()):
            main()

        self.assertEqual(calculate_periods.call_count, 3)
        self.assertEqual(
            retrain.call_args_list,
            [
                call(
                    product="A",
                    population="P1",
                    train_yms=["A01"],
                    backtest_ym="202506",
                    edition="v1",
                    write_db=False,
                ),
                call(
                    product="A",
                    population="P2",
                    train_yms=["A01"],
                    backtest_ym="202506",
                    edition="v1",
                    write_db=False,
                ),
                call(
                    product="B",
                    population="P3",
                    train_yms=["B01"],
                    backtest_ym="202506",
                    edition="v1",
                    write_db=False,
                ),
            ],
        )
