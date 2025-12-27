from src.price_loader import download_stock_prices, compute_monthly_returns
from src.data_loader import load_data
from src.models import train_all_models, compare_with_without_esg


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

    print("\n=== 5) ESG Value Test (A/B) ===")
    compare_with_without_esg(df)


if __name__ == "__main__":
    main()
