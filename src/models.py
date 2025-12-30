import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_loader import prepare_features


# ==========================================================
# PREPROCESSING PIPELINE (pour train_all_models)
# ==========================================================
def build_preprocessor():
    numeric = ["esg", "e", "s", "g",
               "ret_lag_1", "ret_lag_2", "ret_lag_3",
               "ret_lag_4", "ret_lag_5", "ret_lag_6"]
    categorical = ["sector", "industry"]

    return ColumnTransformer([
        ("num", MinMaxScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


# ==========================================================
# EVALUATION
# ==========================================================
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2}


# ==========================================================
# TRAINING ON ALL MODELS (pipeline complet)
# ==========================================================
def train_all_models(df, chronological=False):
    X, y = prepare_features(df)

    if chronological:
        df = df.sort_values("date")
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    preprocessor = build_preprocessor()

    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso(alpha=1e-4)),
        ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    ]

    results = []
    for name, m in models:
        pipe = Pipeline([("prep", preprocessor), ("model", m)])
        results.append(evaluate_model(name, pipe, X_train, X_test, y_train, y_test))

    print("\n=== Model Ranking ===")
    for r in sorted(results, key=lambda x: x["rmse"]):
        print(f"{r['name']} → RMSE={r['rmse']:.4f} | R²={r['r2']:.4f}")

    return results


# ==========================================================
# UTIL: split UNIQUE (indices) pour A/B valid
# ==========================================================
def _make_train_test_indices(df, test_size=0.2, chronological=False, random_state=42):
    if chronological and "date" in df.columns:
        df_sorted = df.sort_values("date")
        split = int(len(df_sorted) * (1 - test_size))
        train_idx = df_sorted.index[:split]
        test_idx = df_sorted.index[split:]
    else:
        train_idx, test_idx = train_test_split(
            df.index, test_size=test_size, random_state=random_state
        )
    return train_idx, test_idx


# ==========================================================
# ESG vs NO-ESG A/B COMPARISON (GLOBAL) - CORRIGÉ
# (même split => Δ comparable)
# ==========================================================
def compare_with_without_esg(df, chronological=False):
    """
    Baseline: lags uniquement
    Full: lags + ESG(num) + sector/industry(cat)
    IMPORTANT: même split baseline/full (corrigé)
    """

    lag_cols = ["ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5", "ret_lag_6"]
    esg_num = ["esg", "e", "s", "g"]
    cat_cols = ["sector", "industry"]

    needed = lag_cols + esg_num + cat_cols + ["return"]
    df2 = df.dropna(subset=needed).copy()

    train_idx, test_idx = _make_train_test_indices(df2, test_size=0.2, chronological=chronological)

    y_train = df2.loc[train_idx, "return"]
    y_test = df2.loc[test_idx, "return"]

    Xb_train = df2.loc[train_idx, lag_cols]
    Xb_test = df2.loc[test_idx, lag_cols]

    Xf_train = df2.loc[train_idx, lag_cols + esg_num + cat_cols]
    Xf_test = df2.loc[test_idx, lag_cols + esg_num + cat_cols]

    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=1e-4),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []

    for name, model in models.items():
        base = Pipeline([("scale", MinMaxScaler()), ("model", model)])

        full = Pipeline([
            ("prep", ColumnTransformer([
                ("num", MinMaxScaler(), lag_cols + esg_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ])),
            ("model", model)
        ])

        r2_base = r2_score(y_test, base.fit(Xb_train, y_train).predict(Xb_test))
        r2_full = r2_score(y_test, full.fit(Xf_train, y_train).predict(Xf_test))

        rows.append([name, r2_base, r2_full, r2_full - r2_base])

    df_out = pd.DataFrame(rows, columns=["Model", "R² Baseline", "R² With ESG(+sector/industry)", "Δ Gain"])

    print("\n========== ESG CONTRIBUTION ANALYSIS (GLOBAL, CORRIGÉ) ==========")
    print(df_out.to_string(index=False))
    print("---------------------------------------------------------------")
    print("Δ > 0 → ajout ESG(+sector/industry) améliore la prédiction")
    print("Δ < 0 → ajout apporte peu / dégrade out-of-sample")

    return df_out


# ==========================================================
# ESG NUMERIC ONLY (GLOBAL)
# pour séparer effet ESG (scores) vs effet secteur/industrie
# ==========================================================
def compare_esg_numeric_only(df, chronological=False):
    """
    Baseline: lags uniquement
    Full: lags + ESG (esg,e,s,g) uniquement
    => mesure plus “pure” de l’effet ESG numérique
    """

    lag_cols = ["ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5", "ret_lag_6"]
    esg_num = ["esg", "e", "s", "g"]

    needed = lag_cols + esg_num + ["return"]
    df2 = df.dropna(subset=needed).copy()

    train_idx, test_idx = _make_train_test_indices(df2, test_size=0.2, chronological=chronological)

    y_train = df2.loc[train_idx, "return"]
    y_test = df2.loc[test_idx, "return"]

    Xb_train = df2.loc[train_idx, lag_cols]
    Xb_test = df2.loc[test_idx, lag_cols]

    Xf_train = df2.loc[train_idx, lag_cols + esg_num]
    Xf_test = df2.loc[test_idx, lag_cols + esg_num]

    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=1e-4),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    for name, model in models.items():
        base = Pipeline([("scale", MinMaxScaler()), ("model", model)])
        full = Pipeline([("scale", MinMaxScaler()), ("model", model)])

        r2_base = r2_score(y_test, base.fit(Xb_train, y_train).predict(Xb_test))
        r2_full = r2_score(y_test, full.fit(Xf_train, y_train).predict(Xf_test))

        rows.append([name, r2_base, r2_full, r2_full - r2_base])

    df_out = pd.DataFrame(rows, columns=["Model", "R² Baseline", "R² With ESG(numeric)", "Δ Gain"])

    print("\n========== ESG NUMERIC ONLY (GLOBAL) ==========")
    print(df_out.to_string(index=False))
    print("------------------------------------------------")
    print("Δ > 0 → les scores ESG (numériques) aident")
    print("Δ < 0 → peu/pas d’info ESG numérique")

    return df_out


# ==========================================================
# ESG vs NO-ESG A/B COMPARISON (BY SECTOR)
# ==========================================================
def compare_esg_by_sector(df, min_rows=500, chronological=False):
    """
    Compare baseline vs with-ESG *par secteur*.
    Baseline: lags uniquement
    Full: lags + ESG(num) + industry(cat)
    (sector est constant dans un sous-ensemble, donc inutile en feature)
    """

    lag_cols = ["ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5", "ret_lag_6"]
    esg_num = ["esg", "e", "s", "g"]
    cat_cols = ["industry"]

    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=1e-4),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []

    for sector, sub in df.groupby("sector"):
        needed = lag_cols + esg_num + cat_cols + ["return"]
        sub = sub.dropna(subset=needed).copy()

        if len(sub) < min_rows:
            continue

        train_idx, test_idx = _make_train_test_indices(
            sub, test_size=0.2, chronological=chronological, random_state=42
        )

        y_train = sub.loc[train_idx, "return"]
        y_test = sub.loc[test_idx, "return"]

        Xb_train = sub.loc[train_idx, lag_cols]
        Xb_test = sub.loc[test_idx, lag_cols]

        Xf_train = sub.loc[train_idx, lag_cols + esg_num + cat_cols]
        Xf_test = sub.loc[test_idx, lag_cols + esg_num + cat_cols]

        for name, model in models.items():
            base = Pipeline([("scale", MinMaxScaler()), ("model", model)])

            full = Pipeline([
                ("prep", ColumnTransformer([
                    ("num", MinMaxScaler(), lag_cols + esg_num),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ])),
                ("model", model)
            ])

            r2_base = r2_score(y_test, base.fit(Xb_train, y_train).predict(Xb_test))
            r2_full = r2_score(y_test, full.fit(Xf_train, y_train).predict(Xf_test))

            rows.append([sector, name, len(sub), r2_base, r2_full, r2_full - r2_base])

    out = pd.DataFrame(rows, columns=[
        "Sector", "Model", "N", "R² Baseline", "R² With ESG", "Δ ESG Gain"
    ])

    if not out.empty:
        print("\n========== ESG GAIN PAR SECTEUR ==========")
        pivot = out.pivot_table(index="Sector", columns="Model", values="Δ ESG Gain", aggfunc="mean")
        print(pivot.sort_index().to_string())
        print("-----------------------------------------")
        print("Δ > 0 → ESG améliore la prédiction dans ce secteur")

    return out
