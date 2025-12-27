import pandas as pd
import yfinance as yf
import time
import os


input_xlsx = r"C:\Users\ish4q\Desktop\ESG Prediction Project\data\raw\Ishac.xlsx"
tidy_output = r"C:\Users\ish4q\Desktop\ESG Prediction Project\data\processed\Ishac_tidy.csv"
final_output = r"C:\Users\ish4q\Desktop\ESG Prediction Project\data\processed\Ishac_tidy_sector_industry.csv"

df = pd.read_excel(input_xlsx, engine="openpyxl")

df = df.rename(columns={
    "TICKER SYMBOL": "ticker",
    "ISIN CODE": "isin",
    "NAME": "name",
    "SCORE": "score_text"
})

print("\n=== Loading ESG dataset ===")
print("Columns:", df.columns.tolist())
print("Raw shape:", df.shape)

# --- Detect block size: number of companies ---
num_companies = df["ticker"].notna().sum()
print("Detected companies:", num_companies)

if len(df) < 4 * num_companies:
    raise ValueError("File doesn't appear to contain 4 blocks (ESG / E / S / G).")

blocks = []
for i in range(4):
    block = df.iloc[i*num_companies:(i+1)*num_companies].copy()
    block["block_id"] = i
    blocks.append(block)

# --- Identify block type ---
def detect_block_type(block):
    for v in block["score_text"]:
        if pd.notna(v):
            text = str(v).lower()
            if "esg score" in text: return "ESG"
            if "governance" in text: return "G"
            if "social" in text: return "S"
            if "environment" in text: return "E"
    return None

block_types = {}
for b in blocks:
    btype = detect_block_type(b)
    print(f"Block {b['block_id'].iloc[0]} detected as:", btype)
    block_types[btype] = b

expected_types = ["ESG","E","S","G"]

# --- Year columns ---
year_cols = [c for c in df.columns if str(c).isdigit() or isinstance(c,int)]
print("Year columns:", year_cols)

# --- Convert block to long format ---
def block_to_long(block, metric):
    esg_block = block_types["ESG"].reset_index(drop=True)
    block = block.reset_index(drop=True)

    block["ticker"] = esg_block["ticker"]
    block["name"] = esg_block["name"]

    for c in year_cols:
        block[c] = block[c].astype(str).str.replace(",", ".", regex=False).replace("nan", pd.NA)
        block[c] = pd.to_numeric(block[c], errors="coerce")

    long_df = block.melt(id_vars=["ticker","name"], value_vars=year_cols,
                         var_name="year", value_name=metric)
    long_df["year"] = long_df["year"].astype(int)
    return long_df

long_data = {}

for t in expected_types:
    if t in block_types:
        long_data[t] = block_to_long(block_types[t], t)
        print(f"{t} long format shape:", long_data[t].shape)
    else:
        long_data[t] = pd.DataFrame(columns=["ticker","name","year",t])

# --- Merge all scores ---
final = long_data["ESG"]
for t in ["E","S","G"]:
    final = final.merge(long_data[t], on=["ticker","name","year"], how="outer")

final = final.sort_values(["ticker","year"]).reset_index(drop=True)

os.makedirs(os.path.dirname(tidy_output), exist_ok=True)
final.to_csv(tidy_output, index=False, float_format="%.2f")

print("\n📁 Saved tidy dataset:", tidy_output)
print("Companies:", final["ticker"].nunique())



print("\n=== Fetching Yahoo Finance sector & industry ===")

tickers = final["ticker"].dropna().unique()
print("Tickers to fetch:", len(tickers))

info_map = {}

for t in tickers:
    try:
        data = yf.Ticker(t).info
        info_map[t] = {
            "sector": data.get("sector"),
            "industry": data.get("industry")
        }
        print(f"{t} → Sector: {info_map[t]['sector']} | Industry: {info_map[t]['industry']}")
    except:
        print(f"{t} ❌ Failed to fetch")

    time.sleep(0.25)  # anti rate-limit

df_info = pd.DataFrame.from_dict(info_map, orient="index").reset_index()
df_info = df_info.rename(columns={"index":"ticker"})

df_final = final.merge(df_info, on="ticker", how="left")
df_final.to_csv(final_output, index=False)

print("\n🎉 Final dataset created with ESG + Sector + Industry")
print("→", final_output)
print("\nPreview:\n", df_final.head(10))
