import pandas as pd
import numpy as np

csv_df = pd.read_csv('files/example_csv.csv', header=None)
json_df = pd.read_json('files/example_json.json', header=None)
parquet_df = pd.read_parquet('files/example.parquet')
feather_df = pd.read_feather('files/example.feather')


# clean string columns
def clean_string_col(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        country=df["country"]
        .str.strip()
        .str.upper()
        .replace({"USA": "US", "UNITED STATES": "US"})
    )

# deal with missing values
def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        numeric_col=df["numeric_col"].fillna(df["numeric_col"].median()),
        numeric_col2=df["numeric_col2"].fillna(0)
    )

# feature engineering / new columns
def add_new_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        new_col=df["colA"] + 2*df["colB"],
        boolean_is_senior=df["age"] >= 65,
    )

# sanity checks on data. if fail, entire pipe ends with AssertionError
def validate(df: pd.DataFrame) -> pd.DataFrame:
    assert df["age"].between(0, 120).all()
    return df

# add datetime features
def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        date=lambda d: pd.to_datetime(d["date"]),
        weekday=lambda d: d["date"].dt.day_name(),
    )

# binning/buckets
def add_age_buckets(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        age_buckets=lambda d: pd.cut(d["age"], bins=[0, 16, 65])
    )


df_final = (
    df
    .filter(['colA', 'colB', 'colC'])  # select columns
    .query('colC > 0')                 # filter rows
    .dropna(subset=['colA', 'colB'])   # drop rows with missing values in colA or colB
    .drop_duplicates(subset=['id'])    # remove duplicate rows
    .merge(                            # left-join (merge safer than 'join')
        other=other_df,
        how='left',
        on='id'
    )
    .pipe(add_age_buckets)
    .pipe(clean_string_col)
    .pipe(impute_values)
    .pipe(add_new_cols)

    #datetime stuff
    .pipe(add_datetime_features)
    .set_index("timestamp")
    .loc["2024-01-01":"2024-01-07"] # this can only be done bc index is datetime

    .pipe(validate)

    .groupby('group', as_index=False)  # calculate statistics per group
    .agg(
        sum_per_group = ('summands', "sum"),
        avg_per_group = ('numeric_col', "mean"),
        count_per_group = ('id', "count"),
        unique_per_group = ('category', "nunique")
    )
    .sort_values('numeric_col', ascending=False)  # sort
)