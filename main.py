"""Project command overview."""
# 告訴你有哪些指令可以用


def main() -> None:
    """Print the independent pipeline entry points."""
    print("Available independent commands:")
    # 建立資料表
    # 更新特徵資料
    print("  uv run mlops-refresh-features --ym YYYYMM [--groups GROUP ...]")
    # 更新母體資料
    print("  uv run mlops-refresh-population --as-of-ym YYYYMM --product PRODUCT")
    # 抽樣產生訓練母體
    print("  uv run mlops-refresh-training-population --ym YYYYMM [--product PRODUCT ...]")
    # 執行模型訓練
    print(
        "  uv run mlops-retrain --product PRODUCT --population POPULATION "
        "--edition VERSION [--as-of-ym YYYYMM] [--train-yms YYYYMM ...] [--backtest-ym YYYYMM]"
    )
    print("    (If dates not specified, auto-calculated from product schedule)")
    # 執行預測
    print("  uv run mlops-predict --product PRODUCT --population POPULATION --ym YYYYMM --edition VERSION")


if __name__ == "__main__":
    main()
