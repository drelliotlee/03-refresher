import pandas as pd
import numpy as np

csv_df = pd.read_csv('files/example_csv.csv', header=None)
json_df = pd.read_json('files/example_json.json', header=None)
parquet_df = pd.read_parquet('files/example.parquet')
feather_df = pd.read_feather('files/example.feather')


# cleaning: messy string columns
def clean_string_col(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        country=df["country"]
        .str.strip()
        .str.upper()
        .replace({"USA": "US", "UNITED STATES": "US"})
    )

# cleaning: improper missing values
def proper_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_tokens = {"", "NA", "NaN", "null", "UNKNOWN"}
    return df.assign(
        col_name=df["col_name"].astype("string").str.strip().replace(missing_tokens, pd.NA)
    )

# cleaning: deal with missing values
def cleaning_impute_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        numeric_col=df["numeric_col"].fillna( df["numeric_col"].median() ),
        numeric_col2=df["numeric_col2"].fillna( 0 )
    )

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

# sanity checks on data. if fail, entire pipe ends with AssertionError
def validate(df: pd.DataFrame) -> pd.DataFrame:
    assert df["age"].between(0, 120).all()
    return df


# Method 1: groupby + agg (collapses to one row per group)
df_aggregated = (
    df
    .filter(['colA', 'colB', 'colC'])  # select columns 
    .query('colC > 0')                 # filter rows
    .merge(                            # joining: (merge safer than 'join')
        other=other_df,
        how='left',
        on='id'
    )
    .drop_duplicates(subset=['id'])    # cleaning: duplicate rows? drop rows
    .pipe(proper_missing_values)       # cleaning: improper missing values -> pd.NA
    .pipe(clean_string_col)            # cleaning: messy string columns
    .pipe(cleaning_impute_values)      # cleaning: missing values? impute
    .dropna(subset=['colA', 'colB'])   # cleaning: missing values? drop rows

    .pipe(fe_add_new_cols)             # feature engineering : 2 simple new columns
    .pipe(fe_add_age_buckets)          # feature engineering: binning/buckets
    
    # Method 1: calculate statistics per group (collapses many rows into one row per group)
    .groupby('group', as_index=False)  
    .agg(
        sum_per_group = ('numeric_col', "sum"),
        avg_per_group = ('numeric_col', "mean"),
        count_per_group = ('id_col', "count"),
        unique_per_group = ('category_col', "nunique")
    )

    # Method 2:calculate statistics per group (keeps original rows)
    .assign(
        sum_per_group=lambda d: d.groupby('group')['numeric_col'].transform('sum'),
        avg_per_group=lambda d: d.groupby('group')['numeric_col'].transform('mean'),
        count_per_group=lambda d: d.groupby('group')['id_col'].transform('count'),
        unique_per_group=lambda d: d.groupby('group')['category_col'].transform('nunique')
    )

    .sort_values('avg_per_group', ascending=False)  # sorting rows
)

# Save locally and upload to S3
df_aggregated.to_parquet('output/processed_data.parquet', index=False)
df_aggregated.to_parquet('s3://my-bucket/processed_data.parquet', index=False)
