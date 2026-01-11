import yfinance as yf
import pandas as pd
from pathlib import Path


class PriceDataLoader:
    """
    Dowloads stock price data from Yahoo Finance and computes monthly returns.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        verbose: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.verbose = verbose

        self.prices_path = self.raw_dir / "stock_prices.csv"
        self.returns_path = self.processed_dir / "monthly_returns.csv"
        self.esg_path = self.processed_dir / "sp500_esg_risk_ratings.csv"

    # --------------------------------------------------
    # ESG tickers
    # --------------------------------------------------
    def load_esg_tickers(self) -> list[str]:
        df = pd.read_csv(self.esg_path)
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        return df["ticker"].unique().tolist()

    # --------------------------------------------------
    # Internal utils
    # --------------------------------------------------
    @staticmethod
    def _flatten_yf_dataframe(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(ticker, axis=1, level=-1)
            except Exception:
                data = data.droplevel(0, axis=1)

        data.columns = [str(c) for c in data.columns]
        return data

    # --------------------------------------------------
    # Download prices
    # --------------------------------------------------
    def download_stock_prices(self, start="2020-01-01", end="2024-01-01") -> pd.DataFrame:
        tickers = self.load_esg_tickers()

        if self.verbose:
            print(f"Downloading data for {len(tickers)} tickers...")

        valid, failed, all_series = [], [], []

        for t in tickers:
            try:
                df = yf.download(
                    t,
                    start=start,
                    end=end,
                    interval="1mo",
                    auto_adjust=True,
                    progress=False,
                )

                if df.empty:
                    failed.append(t)
                    continue

                df = self._flatten_yf_dataframe(df, t)

                price_col = "Adj Close" if "Adj Close" in df else "Close"
                series = df[price_col].rename(t)

                all_series.append(series)
                valid.append(t)

            except Exception:
                failed.append(t)

        prices = pd.concat(all_series, axis=1) if all_series else pd.DataFrame()

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        prices.to_csv(self.prices_path)

        if self.verbose:
            print("Prices saved → data/raw/stock_prices.csv")
            print(f"Valid tickers : {len(valid)}")
            print(f"Failed tickers: {len(failed)}")

        return prices

    # --------------------------------------------------
    # Monthly returns
    # --------------------------------------------------
    def compute_monthly_returns(self) -> pd.DataFrame:
        if not self.prices_path.exists():
            if self.verbose:
                print(" No price data available. Cannot compute returns.")
            return pd.DataFrame()

        df = pd.read_csv(self.prices_path, index_col=0, parse_dates=True)

        if df.empty:
            if self.verbose:
                print(" Price file empty. Cannot compute returns.")
            return pd.DataFrame()

        df_m = df.resample("ME").last()
        ret = df_m.pct_change().dropna(how="all")

        df_long = (
            ret.reset_index()
            .melt(id_vars="Date", var_name="ticker", value_name="return")
            .rename(columns={"Date": "date"})
            .dropna(subset=["return"])
        )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df_long.to_csv(self.returns_path, index=False)

        if self.verbose:
            print("Monthly returns saved → data/processed/monthly_returns.csv")
            print(f"Rows: {len(df_long)}")

        return df_long
