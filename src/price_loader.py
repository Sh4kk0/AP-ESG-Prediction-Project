import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ----------- read ESG tickers (no reprocessing needed now) -----------

def load_esg_tickers():
    """
    Lecture directe du fichier déjà nettoyé :
    sp500_esg_risk_ratings.csv contient : ticker,name,year,esg,e,s,g,sector,industry
    On extrait uniquement les tickers uniques.
    """
    df = pd.read_csv(PROCESSED_DIR / "sp500_esg_risk_ratings.csv")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df["ticker"].unique().tolist()


# ----------- utilitaire pour MultiIndex yfinance -----------

def _flatten_yf_dataframe(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(ticker, axis=1, level=-1)
        except Exception:
            data = data.droplevel(0, axis=1)

    data.columns = [str(c) for c in data.columns]
    return data


# ----------- download des prix -----------

def download_stock_prices(start="2020-01-01", end="2024-01-01"):
    tickers = load_esg_tickers()
    print(f"Downloading data for {len(tickers)} tickers...")

    valid, failed, all_series = [], [], []

    for t in tickers:
        try:
            df = yf.download(
                t, start=start, end=end,
                interval="1mo", auto_adjust=True,
                progress=False
            )

            if df.empty:
                failed.append(t); continue

            df = _flatten_yf_dataframe(df, t)

            price_col = "Adj Close" if "Adj Close" in df else "Close"
            series = df[price_col].rename(t)
            all_series.append(series)
            valid.append(t)

        except Exception:
            failed.append(t)

    if all_series:
        prices = pd.concat(all_series, axis=1)
    else:
        prices = pd.DataFrame()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(RAW_DIR / "stock_prices.csv")

    print(f"\n📁 Prices saved → data/raw/stock_prices.csv")
    print(f"Valid tickers : {len(valid)}")
    print(f"Failed tickers: {len(failed)}\n")

    return prices


# ----------- returns mensuels -----------

def compute_monthly_returns():
    df = pd.read_csv(RAW_DIR / "stock_prices.csv", index_col=0, parse_dates=True)

    if df.empty:
        print("❗ No price data available. Cannot compute returns.")
        return pd.DataFrame()

    df_m = df.resample("M").last()
    ret = df_m.pct_change().dropna(how="all")

    df_long = ret.reset_index().melt(
        id_vars="Date", var_name="ticker", value_name="return"
    ).rename(columns={"Date": "date"}).dropna(subset=["return"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(PROCESSED_DIR / "monthly_returns.csv", index=False)

    print(f"📁 Monthly returns saved → data/processed/monthly_returns.csv")
    print(f"Rows:", len(df_long))
    return df_long

