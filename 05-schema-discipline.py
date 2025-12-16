import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, None],          # integers + missing
        "age": [25, 40, 120, None, 30],         # fits in Int8 but pandas won't guess
        "is_active": ["True", "False", None, "True", "FALSE"],  # boolean swamp
        "country": ["US", "CA", "US", "MX", "CA"],              # low-cardinality strings
        "score": [98.2, 87.5, 91.0, 88.0, 92.3], # float64 by default
    }
)

# -----------------------------------------------
# BEST PRACTICE 1: enforce dtypes at import time
# -----------------------------------------------

df2 = pd.read_csv(
    'files/example.csv',
    dtype={
        "user_id": "Int64",
        "age": "Int8",
        "is_active": "boolean",
        "country": "category",
        "score": "float32",
    }
)

df3 = pd.read_parquet(
    'files/example.parquet',
    dtype={
        "user_id": "Int64",
        "age": "Int8",
        "is_active": "boolean",
        "country": "category",
        "score": "float32",
    }
)


# -----------------------------------------------
# discipline w booleans
# -----------------------------------------------

# bool can only be True False, cant deal with missing values
# boolean (dtype from pandas) can hold NaNs
def clean_is_active_boolean(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        is_active=(
            df["is_active"]
            .str.upper()
            .map({"TRUE": True, "FALSE": False})
            .astype("boolean")
        )
    )

# -----------------------------------------------
# discipline w categorical cols
# -----------------------------------------------

# category uses less memory than strings
def cast_country_to_category(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        country=df["country"].astype("category")
    )

# freeze categories
COUNTRY_DOMAIN = ["US", "CA", "MX"]
def freeze_country_domain(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        country=df["country"].cat.set_categories(COUNTRY_DOMAIN)
    )

# -----------------------------------------------
# discipline w numeric cols
# -----------------------------------------------

# int64 cant hold NaNs, use Int64 instead
def cast_user_id_to_Int64(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        user_id=df["user_id"].astype("Int64")
    )

# as long as below 128, Int8 is sufficient
def downcast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        age=df["age"].astype("Int8"),          # valid range: -128..127
        score=df["score"].astype("float32"),   # model-safe precision
    )

# -----------------------------------------------
# BEST PRACTICE 3: enforce schema again at end of pipeline
# -----------------------------------------------

# why at the end? many operations quietly coerce dtypes
# - merge creates NaNs, that changes integers into floats
# - concat upcasts dtypes to accommodate all inputs
# - string operations may change categories back to object
# - fillna on integer cols makes them floats
SCHEMA = {
    "user_id": "Int64",
    "age": "Int8",
    "is_active": "boolean",
    "country": "category",
    "score": "float32",
}

# if schema is violated, entire pipe fails
def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        **{
            col: df[col].astype(dtype)
            for col, dtype in SCHEMA.items()
        }
    )


# -----------------------------------------------
# FINAL PIPELINE
# -----------------------------------------------

df_final = (
    df
    .pipe(cast_user_id_to_Int64)
    .pipe(clean_is_active_boolean)
    .pipe(cast_country_to_category)
    .pipe(downcast_numerics)
    .pipe(freeze_country_domain)
    .pipe(enforce_schema)
)