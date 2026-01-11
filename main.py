from src.data_loader import ESGDataLoader
from src.models import ESGModelExperiment


def main():
    # Load dataset final
    loader = ESGDataLoader(n_lags=6)
    df = loader.load_data()

    print(f"\nFinal dataset size → {len(df)} rows")
    print(df.head())

    # ML experiment
    exp = ESGModelExperiment(chronological=False)
    exp.split(df)

    print("\n=== 4) Train ML Models ===")
    exp.train_all()

    print("\n=== 5) ESG Value Test ===")
    exp.ab_global()

    print("\n=== 6) ESG Value by sector ===")
    exp.ab_by_sector(min_rows=200)


if __name__ == "__main__":
    main()
