import pandas as pd
import numpy as np

csv_df = pd.read_csv('files/example_csv.csv', header=None)
json_df = pd.read_json('files/example_json.json', header=None)
parquet_df = pd.read_parquet('files/example.parquet')
feather_df = pd.read_feather('files/example.feather')


# select columns
def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.filter(['colA', 'colB', 'colC'])

# filter rows
def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.query('colC > 0')

# joining
def merge_other_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(other=other_df, how='left', on='id')

# cleaning: duplicate rows
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=['id'])

# cleaning: improper missing values
def proper_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_tokens = {"", "NA", "NaN", "null", "UNKNOWN"}
    return df.assign(
        col_name=df["col_name"].astype("string").str.strip().replace(missing_tokens, pd.NA)
    )

# cleaning: messy string columns
def clean_string_col(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        country=df["country"]
        .str.strip()
        .str.upper()
        .replace({"USA": "US", "UNITED STATES": "US"})
    )

# cleaning: deal with missing values
def cleaning_impute_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        numeric_col=df["numeric_col"].fillna( df["numeric_col"].median() ),
        numeric_col2=df["numeric_col2"].fillna( 0 )
    )

# cleaning: drop rows with missing values
def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['colA', 'colB'])

# feature engineering : 2 simple new columns
def fe_add_new_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        new_col=df["colA"] + 2*df["colB"],
        boolean_is_senior=df["age"] >= 65,
    )

# feature engineering: binning/buckets
def fe_add_age_buckets(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        age_buckets=pd.cut(df["age"], bins=[0, 16, 65, 120])
    )

# Method 1 :calculate statistics per group (combines rows)
def aggregate_by_group(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby('group', as_index=False).agg(
        sum_per_group = ('numeric_col', "sum"),
        avg_per_group = ('numeric_col', "mean"),
        count_per_group = ('id_col', "count"),
        unique_per_group = ('category_col', "nunique")
    )
    return result

# Method 2: calculate statistics per group (keeps original rows)
def add_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.assign(
        sum_per_group=lambda d: d.groupby('group')['numeric_col'].transform('sum'),
        avg_per_group=lambda d: d.groupby('group')['numeric_col'].transform('mean'),
        count_per_group=lambda d: d.groupby('group')['id_col'].transform('count'),
        unique_per_group=lambda d: d.groupby('group')['category_col'].transform('nunique')
    )
    return result

# sorting rows
def sort_by_avg(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values('avg_per_group', ascending=False)

# save output
def save_output(df: pd.DataFrame) -> pd.DataFrame:
    df.to_parquet('output/processed_data.parquet', index=False)
    df.to_parquet('s3://my-bucket/processed_data.parquet', index=False)
    return df


# Complete pipeline using .pipe() for all steps
df_aggregated = (
    df
    .pipe(select_columns)
    .pipe(filter_rows)
    .pipe(merge_other_df)
    .pipe(remove_duplicates)
    .pipe(proper_missing_values)
    .pipe(clean_string_col)
    .pipe(cleaning_impute_values)
    .pipe(drop_missing_rows)
    .pipe(fe_add_new_cols)
    .pipe(fe_add_age_buckets)
    .pipe(aggregate_by_group)
    .pipe(add_group_statistics)
    .pipe(sort_by_avg)
    .pipe(save_output)
)
