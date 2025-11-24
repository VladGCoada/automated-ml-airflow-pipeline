"""
Full preprocessing for the M5 sales dataset.

Creates a long-format dataframe with:
- item_id, store_id, dept_id, cat_id
- date
- demand
- price features (selling_prices.csv)
- calendar features (calendar.csv)
- lag features
- rolling windows (7, 28)
- day-of-week, event flags

Output: /opt/airflow/data/processed_m5.parquet
"""

import os
import pandas as pd
import numpy as np


DATA_DIR = "/opt/airflow/data"
OUTPUT_PATH = f"{DATA_DIR}/processed_m5.parquet"


def load_raw():
    sales = pd.read_csv(f"{DATA_DIR}/sales_train_evaluation.csv")
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
    prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
    return sales, calendar, prices


def melt_sales(sales):
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_vars = [col for col in sales.columns if col.startswith("d_")]

    long_df = sales.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="d",
        value_name="demand"
    )
    return long_df


def merge_calendar(long_df, calendar):
    calendar = calendar.rename(columns={"d": "d"})
    merged = long_df.merge(calendar, on="d", how="left")

    # Rename date column to proper datetime
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def merge_prices(df, prices):
    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )
    return df


def add_time_features(df):
    df["wday"] = df["date"].dt.weekday
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["wday"] >= 5).astype(int)
    return df


def add_lag_features(df):
    df = df.sort_values(["id", "date"])
    df["lag_1"] = df.groupby("id")["demand"].shift(1)
    df["lag_7"] = df.groupby("id")["demand"].shift(7)
    df["lag_28"] = df.groupby("id")["demand"].shift(28)
    return df


def add_rolling_windows(df):
    df["rolling_7"] = df.groupby("id")["demand"].shift(1).rolling(7).mean()
    df["rolling_28"] = df.groupby("id")["demand"].shift(1).rolling(28).mean()
    return df


def main():
    print("[M5 Preprocessing] Loading raw datasets…")
    sales, calendar, prices = load_raw()

    print("[M5 Preprocessing] Melting sales to long format…")
    df = melt_sales(sales)

    print("[M5 Preprocessing] Merging calendar…")
    df = merge_calendar(df, calendar)

    print("[M5 Preprocessing] Merging price data…")
    df = merge_prices(df, prices)

    print("[M5 Preprocessing] Adding time features…")
    df = add_time_features(df)

    print("[M5 Preprocessing] Adding lag features…")
    df = add_lag_features(df)

    print("[M5 Preprocessing] Adding rolling means…")
    df = add_rolling_windows(df)

    print(f"[M5 Preprocessing] Saving to {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)

    print("[M5 Preprocessing] DONE!")


if __name__ == "__main__":
    main()

