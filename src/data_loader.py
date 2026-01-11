import pandas as pd
from pathlib import Path


class ESGDataLoader:
    """
    Responsable du chargement et de la préparation des données ESG + returns.
    - charge les CSV
    - merge ESG / returns
    - ajoute les lags
    """

    def __init__(self, processed_dir: str | Path = "data/processed", n_lags: int = 6):
        self.processed_dir = Path(processed_dir)
        self.n_lags = n_lags

        self.esg_path = self.processed_dir / "sp500_esg_risk_ratings.csv"
        self.returns_path = self.processed_dir / "monthly_returns.csv"
        self.final_path = self.processed_dir / "final_dataset.csv"

    # --------------------------------------------------
    # Loaders
    # --------------------------------------------------
    def load_esg(self) -> pd.DataFrame:
        esg = pd.read_csv(self.esg_path)
        esg["ticker"] = esg["ticker"].astype(str).str.upper().str.strip()
        esg["year"] = esg["year"].astype(int)
        return esg

    def load_returns(self) -> pd.DataFrame:
        ret = pd.read_csv(self.returns_path)
        ret["date"] = pd.to_datetime(ret["date"], errors="coerce")
        ret = ret.dropna(subset=["date"])
        ret["ticker"] = ret["ticker"].astype(str).str.upper().str.strip()
        ret["year"] = ret["date"].dt.year.astype(int)
        return ret

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    def add_return_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ticker", "date"]).copy()
        g = df.groupby("ticker")["return"]
        for i in range(1, self.n_lags + 1):
            df[f"ret_lag_{i}"] = g.shift(i)
        return df

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def load_data(self, save: bool = True, verbose: bool = True) -> pd.DataFrame:
        esg = self.load_esg()
        ret = self.load_returns()

        if verbose:
            print(f"ESG rows: {len(esg)}, Returns rows: {len(ret)}")

        df = ret.merge(esg, on=["ticker", "year"], how="inner")

        df = self.add_return_lags(df)

        # Drop uniquement ce qui est nécessaire
        required = (
            ["return"]
            + [f"ret_lag_{i}" for i in range(1, self.n_lags + 1)]
            + ["esg", "e", "s", "g", "sector", "industry"]
        )
        df = df.dropna(subset=required)

        if save:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.final_path, index=False)
            if verbose:
                print("📁 Final dataset saved → data/processed/final_dataset.csv")
                print(f"Final dataset rows: {len(df)}")

        return df

    # --------------------------------------------------
    # ML helper
    # --------------------------------------------------
    @staticmethod
    def prepare_features(df: pd.DataFrame):
        numeric = ["esg", "e", "s", "g"] + [f"ret_lag_{i}" for i in range(1, 7)]
        categorical = ["sector", "industry"]

        X = df[numeric + categorical]
        y = df["return"]
        return X, y
