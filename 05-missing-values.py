# GOOD missing values
# np.nan  -  numpy's missing value indicator, prints as nan. 
#            works with float64 // doesn't work with int64
# pd.NA   -  pandas' missing value indicator, prints as <NA>
#            works with Int64, boolean, category // doesn't work with float64

# BAD missing values
# 'NA'    -  string 'NA'
# 'NaN'   -  string 'NaN'
# ''      -  string empty
# 'null'  -  string 'null'
# None    -  python's missing value indicator

# after cleaning, float32 and float64 columns should use np.nan
# after cleaning, Int64, boolean, and category columns should use pd.NA

# convert bad missing values to good missing values
df["name"] = (
    df["name"]
    .str.strip()
    .replace("", pd.NA)
    .replace("NA", pd.NA)
    .replace("NaN", pd.NA)
    .replace("null", pd.NA)
    .replace("UNKNOWN", pd.NA)
)


def normalize_missing_strings(col: str):
    missing_tokens = {"", "NA", "NaN", "null", "UNKNOWN"}
    def normalize_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            **{
                col: (
                    df[col]
                    .astype("string")        # ensures string ops are safe
                    .str.strip()
                    .replace(missing_tokens, pd.NA)
                )
            }
        )

    return _normalize