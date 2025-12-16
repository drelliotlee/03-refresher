import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=100, freq="6h"),
    "category": np.random.choice(["A", "B", "C"], 100),
    "value": np.random.randn(100) * 10 + 50,
    "date_str": ["2024-01-15", "01/20/2024", "2024.02.01", "15-Mar-2024"] * 25,  # messy datetime strings
})


# cast string to datetimes
def parse_datetime_strings(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        date=pd.to_datetime(df["date_str"], format="mixed", dayfirst=False)
    )

# new columns extracting datetime components
def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        year=df["timestamp"].dt.year,
        month=df["timestamp"].dt.month,
        day=df["timestamp"].dt.day,
    )

# new columns based on date arithmetic
def add_date_arithmetic(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        one_week_later=df["timestamp"] + pd.Timedelta(days=7),     # fixed/exact durations
        one_month_later=df["timestamp"] + pd.DateOffset(months=1), # respects calendar quirks
        days_since_start=(df["timestamp"] - df["timestamp"].min()).dt.days,
    )

# resampling aka groupby-like aggregations
def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.set_index("timestamp")                                  # requires datetime index
        .resample("D")                                             # D=day, W=week, M=month, Q=quarter, Y=year
        .agg({"value": ["mean", "sum", "min", "max", "count"]})
        .reset_index()
    )

# timezones (a datetime can be both timezone-naive or timezone-aware)
def handle_timezones(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        timestamp_utc=df["timestamp"].dt.tz_localize("UTC"),  # naive -> aware
        timestamp_ny=df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York"), # convert timezones
    )

# FINAL PIPELINES
df_features = (
    df

    .set_index("timestamp")             # filter rows by datetime index
    .sort_index()                       # requires first setting as index and sorting it
    .loc["2024-01-01":"2024-01-07"]     

    .pipe(extract_datetime_features)    # add cols example 1
    .pipe(add_date_arithmetic)          # add cols example 2
    .query('hour >= 6 and hour <= 18')  # filter rows example 2

    .pipe(resample_to_daily)            # resampling aka groupby-like aggregations
    .pipe(handle_timezones)             # timezone handling
)
