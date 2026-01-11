import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_loader import ESGDataLoader


class ESGModelExperiment:


    def __init__(self, test_size=0.2, random_state=42, chronological=False):
        self.test_size = test_size
        self.random_state = random_state
        self.chronological = chronological

        # columns
        self.lags = [f"ret_lag_{i}" for i in range(1, 7)]
        self.esg = ["esg", "e", "s", "g"]
        self.cats = ["sector", "industry"]

        # models
        self.models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge", Ridge()),
            ("Lasso", Lasso(alpha=1e-4)),
            ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        ]

        self._ready = False
        self._df_ref = None  



    def split(self, df: pd.DataFrame):
        """
        Split data into train/test sets.
        """
        self._df_ref = df

        X, y = ESGDataLoader.prepare_features(df)

        if self.chronological and "date" in df.columns:
            df_sorted = df.sort_values("date")
            cut = int(len(df_sorted) * (1 - self.test_size))
            idx_train = df_sorted.index[:cut]
            idx_test = df_sorted.index[cut:]
        else:
            idx_train, idx_test = train_test_split(
                df.index, test_size=self.test_size, random_state=self.random_state
            )

        self.X_train = X.loc[idx_train]
        self.X_test = X.loc[idx_test]
        self.y_train = y.loc[idx_train]
        self.y_test = y.loc[idx_test]

        self._ready = True

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _full_prep(self):
        return ColumnTransformer([
            ("num", MinMaxScaler(), self.lags + self.esg),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.cats),
        ])

    def _fit_pred(self, pipe: Pipeline, X_train, X_test):
        pipe.fit(X_train, self.y_train)
        return pipe.predict(X_test)

    @staticmethod
    def _metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return rmse, mae, r2

    # --------------------------------------------------
    # 1) Train all models 
    # --------------------------------------------------
    def train_all(self):
        assert self._ready, "Call split(df) first"

        prep = self._full_prep()
        results = []

        for name, model in self.models:
            pipe = Pipeline([
                ("prep", prep),
                ("model", clone(model))
            ])

            pred = self._fit_pred(pipe, self.X_train, self.X_test)
            rmse, mae, r2 = self._metrics(self.y_test, pred)

            print(f"\n--- {name} ---")
            print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

            results.append({"name": name, "rmse": rmse, "mae": mae, "r2": r2})

        print("\n=== Model Ranking ===")
        for r in sorted(results, key=lambda x: x["rmse"]):
            print(f"{r['name']} → RMSE={r['rmse']:.4f} | R²={r['r2']:.4f}")

        return results

    # --------------------------------------------------
    # 2) A/B global: baseline vs ESG enhanced
    # --------------------------------------------------
    def ab_global(self):
        """
        Baseline: lags only
        Full: lags + ESG + sector/industry
        """
        assert self._ready, "Call split(df) first"

        Xb_train = self.X_train[self.lags]
        Xb_test = self.X_test[self.lags]

        Xf_train = self.X_train[self.lags + self.esg + self.cats]
        Xf_test = self.X_test[self.lags + self.esg + self.cats]

        rows = []

        for name, model in self.models:
            base = Pipeline([
                ("scale", MinMaxScaler()),
                ("model", clone(model))
            ])

            full = Pipeline([
                ("prep", self._full_prep()),
                ("model", clone(model))
            ])

            pred_base = self._fit_pred(base, Xb_train, Xb_test)
            pred_full = self._fit_pred(full, Xf_train, Xf_test)

            r2_base = float(r2_score(self.y_test, pred_base))
            r2_full = float(r2_score(self.y_test, pred_full))

            rows.append([name, r2_base, r2_full, r2_full - r2_base])

        out = pd.DataFrame(rows, columns=["Model", "R² Baseline", "R² Full", "Δ Gain"])

        print("\n========== ESG CONTRIBUTION (GLOBAL) ==========")
        print(out.to_string(index=False))

        return out

    # --------------------------------------------------
    # 3) A/B by sector 
    # --------------------------------------------------
    def ab_by_sector(self, min_rows=200):


        assert self._ready, "Call split(df) first"
        assert self._df_ref is not None, "Internal df reference missing"
        assert "sector" in self._df_ref.columns, "Column 'sector' missing in df"

        # Info secteur sur le test global
        df_test = self._df_ref.loc[self.X_test.index, ["sector"]].copy()

        # Features baseline/full sur train/test global
        Xb_train = self.X_train[self.lags]
        Xb_test = self.X_test[self.lags]

        Xf_train = self.X_train[self.lags + self.esg + self.cats]
        Xf_test = self.X_test[self.lags + self.esg + self.cats]

        rows = []

        for model_name, model in self.models:
            # Fit global
            base = Pipeline([("scale", MinMaxScaler()), ("model", clone(model))])
            full = Pipeline([("prep", self._full_prep()), ("model", clone(model))])

            base.fit(Xb_train, self.y_train)
            full.fit(Xf_train, self.y_train)

            pred_base = base.predict(Xb_test)
            pred_full = full.predict(Xf_test)

            # Slice for each sector
            tmp = df_test.copy()
            tmp["y"] = self.y_test.values
            tmp["pb"] = pred_base
            tmp["pf"] = pred_full

            for sector, g in tmp.groupby("sector"):
                if len(g) < min_rows:
                    continue

                r2_b = float(r2_score(g["y"], g["pb"]))
                r2_f = float(r2_score(g["y"], g["pf"]))
                rows.append([sector, model_name, int(len(g)), r2_b, r2_f, r2_f - r2_b])

        out = pd.DataFrame(rows, columns=[
            "Sector", "Model", "N_test", "R² Baseline", "R² Full", "Δ Gain"
        ])

        if out.empty:
            print("\n(No sector had enough test rows to report.)")
            return out

        print("\n========== Δ ESG GAIN PER SECTOR ==========")
        
        pivot = out.pivot_table(index="Sector", columns="Model", values="Δ Gain", aggfunc="mean")
        print(pivot.sort_index().to_string())

        return out
