import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def load_esg():
    """
    Load multi-year ESG dataset already processed from CSV.
    Structure required:
    ticker,year,esg,e,s,g,sector,industry
    """
    esg = pd.read_csv(PROCESSED_DIR / "sp500_esg_risk_ratings.csv")

    # ensure correct dtypes
    esg["ticker"] = esg["ticker"].str.upper().str.strip()
    esg["year"] = esg["year"].astype(int)

    return esg


def load_returns():
    """
    Load monthly returns & attach year column for matching with ESG.
    """
    ret = pd.read_csv(PROCESSED_DIR / "monthly_returns.csv")

    ret["date"] = pd.to_datetime(ret["date"], errors="coerce")  # <- FIX
    ret["ticker"] = ret["ticker"].str.upper().str.strip()
    ret["year"] = ret["date"].dt.year

    return ret


def load_data():
    """
    Final dataset used for modeling.
    Merge monthly returns with ESG using Ticker + Year.
    """

    esg = load_esg()
    ret = load_returns()

    print(f"ESG rows: {len(esg)}, Returns rows: {len(ret)}")

    # multi-year join
    df = ret.merge(esg, on=["ticker", "year"], how="inner")

    df.to_csv(PROCESSED_DIR /"../../data/processed/final_dataset.csv", index=False)
    print("📁 Final dataset saved → data/processed/final_dataset.csv")
    print(f"Final dataset rows: {len(df)}")
    
    df = add_return_lags(df, n_lags=6)  # 6 mois de mémoire
    df = df.dropna()   # indispensable sinon NaN sur premières lignes

    return df


def prepare_features(df):
    numeric = ["esg","e","s","g"] + [f"ret_lag_{i}" for i in range(1,7)]
    categorical = ["sector","industry"]

    X = df[numeric + categorical]
    y = df["return"]
    return X, y



def add_return_lags(df, n_lags=6):
    """Ajoute ret-1, ret-2 ... ret-n pour séries temporelles."""
    df = df.sort_values(["ticker", "date"])
    for i in range(1, n_lags+1):
        df[f"ret_lag_{i}"] = df.groupby("ticker")["return"].shift(i)
    
    return df
