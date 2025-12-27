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
# PREPROCESSING PIPELINE
# ==========================================================
def build_preprocessor():
    numeric = ["esg", "e", "s", "g",
               "ret_lag_1","ret_lag_2","ret_lag_3",
               "ret_lag_4","ret_lag_5","ret_lag_6"]
    categorical = ["sector","industry"]

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
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)

    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")

    return {"name":name,"rmse":rmse,"mae":mae,"r2":r2}


# ==========================================================
# TRAINING ON ALL MODELS
# ==========================================================
def train_all_models(df, chronological=False):

    X,y = prepare_features(df)

    if chronological:
        df = df.sort_values("date")
        split = int(len(df)*0.8)
        X_train,X_test = X[:split],X[split:]
        y_train,y_test = y[:split],y[split:]

    else:
        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42
        )

    preprocessor = build_preprocessor()

    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso(alpha=1e-4)),
        ("Random Forest", RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    ]

    results=[]
    for name,m in models:
        pipe = Pipeline([("prep",preprocessor),("model",m)])
        results.append( evaluate_model(name,pipe,X_train,X_test,y_train,y_test) )

    print("\n=== Model Ranking ===")
    for r in sorted(results,key=lambda x:x["rmse"]):
        print(f"{r['name']} → RMSE={r['rmse']:.4f} | R²={r['r2']:.4f}")

    return results


# ==========================================================
# ESG vs NO-ESG A/B COMPARISON
# ==========================================================
def compare_with_without_esg(df):

    lag_cols = ["ret_lag_1","ret_lag_2","ret_lag_3","ret_lag_4","ret_lag_5","ret_lag_6"]
    esg_cols = ["esg","e","s","g","sector","industry"]

    X_base = df[lag_cols]
    X_full = df[lag_cols + esg_cols]
    y = df["return"]

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_base,y,test_size=0.2,random_state=42)
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_full,y,test_size=0.2,random_state=42)

    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=1e-4),
        "RandomForest": RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    rows=[]

    for name,model in models.items():

        base = Pipeline([("scale",MinMaxScaler()),("model",model)])
        full = Pipeline([
            ("prep",ColumnTransformer([
                ("num",MinMaxScaler(),lag_cols+["esg","e","s","g"]),
                ("cat",OneHotEncoder(handle_unknown="ignore"),["sector","industry"])
            ])),
            ("model",model)
        ])

        r2_base = r2_score(yb_test, base.fit(Xb_train,yb_train).predict(Xb_test))
        r2_full = r2_score(yf_test, full.fit(Xf_train,yf_train).predict(Xf_test))

        rows.append([name,r2_base,r2_full,r2_full-r2_base])

    df_out = pd.DataFrame(rows,columns=["Model","R² Baseline","R² With ESG","Δ ESG Gain"])

    print("\n========== ESG CONTRIBUTION ANALYSIS ==========")
    print(df_out.to_string(index=False))
    print("-----------------------------------------------")
    print("Δ > 0 → ESG améliore la prédiction")
    print("Δ < 0 → ESG apporte peu d'information")

    return df_out
