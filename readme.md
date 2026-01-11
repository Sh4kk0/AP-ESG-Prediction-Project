## 📁 Project Structure

```text
ESG Prediction Project/
│
├── main.py                          # Main monthly pipeline
├── environment.yml                  # Dependencies
│
│
│
├── src/
│   ├── price_loader.py              # Price download + monthly returns
│   ├── data_loader.py               # Merge ESG + returns + lag features
│   ├── models.py                    # Baseline vs ESG training + evaluation
│   └── conv.py                      # Converts xls risk ratings into a csv dataframe
│
├── data/
│   ├── raw/
│   │   ├── stock_prices.csv         # Yahoo Finance price history
│   │   └── sp500_esg.csv            # S&P500 list
│   │
│   └── processed/
│       ├── sp500_esg_risk_ratings.csv
│       ├── monthly_returns.csv
│       └── final_dataset.csv        # Final ML-ready dataset
│
├── results/
│   ├── plots/                       
│   │   └── plots.ipynb              # All charts generated
│   │        
│   ├── metrics/
│   │   └── metrics.txt              # Saved model scores
│   │
│   └── report/                      
│       └── report.pdf               # Notes, interpretations
│

| Model             | RMSE ↓ | MAE ↓  | R² ↓   |
| ----------------- | ------ | ------ | ------ |
| Random Forest     | 0.0913 | 0.0669 | 0.0875 |
| Gradient Boosting | 0.0933 | 0.0695 | 0.0454 |
| Lasso             | 0.0951 | 0.0713 | 0.0095 |
| Ridge             | 0.0955 | 0.0718 | 0.0006 |
| Linear Regression | 0.0955 | 0.0718 | 0.0003 |


requirements : 

pandas
numpy
matplotlib
seaborn
scikit-learn
yfinance
pathlib
joblib
