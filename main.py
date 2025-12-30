from src.price_loader import download_stock_prices, compute_monthly_returns
from src.data_loader import load_data
from src.models import (
    train_all_models,
    compare_with_without_esg,
    compare_esg_numeric_only,
    compare_esg_by_sector
)


def main():
    print("\n=== 1) Download Stock Prices ===")
    download_stock_prices()

    print("\n=== 2) Compute Monthly Returns ===")
    compute_monthly_returns()

    print("\n=== 3) Load Final Dataset (ESG + Returns) ===")
    df = load_data()

    print(f"\nFinal dataset size → {len(df)} rows")
    print(df.head())

    print("\n=== 4) Train ML Models ===")
    train_all_models(df, chronological=False)

    print("\n=== 5) ESG Value Test (A/B - Global, CORRIGÉ) ===")
    compare_with_without_esg(df, chronological=False)

    print("\n=== 5bis) ESG Numeric Only (Global) ===")
    compare_esg_numeric_only(df, chronological=False)

    print("\n=== 6) ESG Value Test (A/B - By Sector) ===")
    compare_esg_by_sector(df, min_rows=500, chronological=False)


if __name__ == "__main__":
    main()
