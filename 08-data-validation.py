import pandas as pd
import numpy as np
from typing import List, Dict, Any

np.random.seed(42)
df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6],
    "age": [25, 40, 150, -5, 30, 28],  # has invalid values
    "email": ["a@b.com", "invalid", "c@d.com", "e@f.com", None, "g@h.com"],
    "salary": [50000, 75000, 90000, 120000, -1000, 85000],  # has invalid
    "country": ["US", "CA", "UK", "US", "MX", "INVALID"],
    "score": [0.8, 0.95, 1.2, 0.5, 0.7, -0.1],  # should be 0-1
})


# Generic validation functions, re-useable across all ML projects
class ValidationError(Exception):
    """Custom exception for validation failures"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed with {len(errors)} error(s):\n" + "\n".join(f"  - {e}" for e in errors))

class DataValidator:  
    def __init__(self):
        self.errors: List[str] = []
    
    def check_not_empty(self, df: pd.DataFrame) -> 'DataValidator':
        """Check if dataframe has at least one row"""
        if df.empty:
            self.errors.append("DataFrame is empty (0 rows)")
        return self
    
    def check_columns_exist(self, df: pd.DataFrame, columns: List[str]) -> 'DataValidator':
        """Check if required columns exist in dataframe"""
        missing = [col for col in columns if col not in df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
        return self
    
    def check_dtype(self, df: pd.DataFrame, col: str, expected_dtype: type) -> 'DataValidator':
        """Check if column has expected data type"""
        if col not in df.columns:
            return self  # skip if column doesn't exist (caught by check_columns_exist)
        if not pd.api.types.is_dtype_equal(df[col].dtype, expected_dtype):
            self.errors.append(
                f"{col}: expected dtype {expected_dtype}, got {df[col].dtype}"
            )
        return self
    
    def check_range(self, df: pd.DataFrame, col: str, min_val: float, max_val: float) -> 'DataValidator':
        """Check if a particular column values are within pre-known valid range"""
        invalid = ~df[col].between(min_val, max_val)
        if invalid.any():
            count = invalid.sum()
            examples = df.loc[invalid, col].head(3).tolist()
            self.errors.append(
                f"{col}: {count} values outside range [{min_val}, {max_val}]. Examples: {examples}"
            )
        return self
    
    def check_no_nulls(self, df: pd.DataFrame, col: str) -> 'DataValidator':
        """Check for null values in a particular column"""
        if df[col].isna().any():
            count = df[col].isna().sum()
            self.errors.append(f"{col}: {count} null values found")
        return self
    
    def check_null_percentage(self, df: pd.DataFrame, col: str, max_percent: float) -> 'DataValidator':
        """Check if null percentage has increased above max threshold"""
        null_pct = (df[col].isna().sum() / len(df)) * 100
        if null_pct > max_percent:
            self.errors.append(
                f"{col}: {null_pct:.1f}% null values (max allowed: {max_percent}%)"
            )
        return self
    
    def check_unique(self, df: pd.DataFrame, col: str) -> 'DataValidator':
        """Check for duplicate values in a particular column"""
        if df[col].duplicated().any():
            count = df[col].duplicated().sum()
            self.errors.append(f"{col}: {count} duplicate values found")
        return self
    
    def check_foreign_key(self, df: pd.DataFrame, col: str, reference_df: pd.DataFrame, ref_col: str) -> 'DataValidator':
        """Check if all values in col exist in reference_df[ref_col]"""
        valid_values = set(reference_df[ref_col].unique())
        invalid = ~df[col].isin(valid_values)
        if invalid.any():
            count = invalid.sum()
            examples = df.loc[invalid, col].unique().tolist()[:3]
            self.errors.append(
                f"{col}: {count} values not found in {ref_col}. Examples: {examples}"
            )
        return self
    
    def check_valid_values(self, df: pd.DataFrame, col: str, valid_values: List[Any]) -> 'DataValidator':
        """Check if all values are in allowed set"""
        invalid = ~df[col].isin(valid_values)
        if invalid.any():
            count = invalid.sum()
            examples = df.loc[invalid, col].unique().tolist()[:3]
            self.errors.append(
                f"{col}: {count} invalid values. Examples: {examples}. Expected: {valid_values}"
            )
        return self
    
    def check_date_range(self, df: pd.DataFrame, col: str, min_date: str, max_date: str) -> 'DataValidator':
        """Check if dates are within valid range"""
        df_dates = pd.to_datetime(df[col], errors='coerce')
        min_dt = pd.to_datetime(min_date)
        max_dt = pd.to_datetime(max_date)
        invalid = ~df_dates.between(min_dt, max_dt)
        if invalid.any():
            count = invalid.sum()
            examples = df.loc[invalid, col].head(3).tolist()
            self.errors.append(
                f"{col}: {count} dates outside range [{min_date}, {max_date}]. Examples: {examples}"
            )
        return self
    
    def check_regex(self, df: pd.DataFrame, col: str, pattern: str, description: str = "pattern") -> 'DataValidator':
        """Check if string column matches regex pattern"""
        invalid = ~df[col].astype(str).str.match(pattern, na=False)
        if invalid.any():
            count = invalid.sum()
            examples = df.loc[invalid, col].head(3).tolist()
            self.errors.append(
                f"{col}: {count} values don't match {description}. Examples: {examples}"
            )
        return self
    
    def check_positive(self, df: pd.DataFrame, col: str) -> 'DataValidator':
        """Check if all values are positive in a particular column"""
        invalid = df[col] <= 0
        if invalid.any():
            count = invalid.sum()
            examples = df.loc[invalid, col].head(3).tolist()
            self.errors.append(
                f"{col}: {count} non-positive values found. Examples: {examples}"
            )
        return self
    
    def validate(self) -> None:
        """Raise exception if any errors were collected"""
        if self.errors:
            raise ValidationError(self.errors)

# Now use DataValidator with explicit columns and rules
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    validator = DataValidator()   
    (validator
        .check_not_empty(df)
        .check_columns_exist(df, ["user_id", "age", "email", "salary", "country", "score"])
        .check_dtype(df, "user_id", np.int64)
        .check_dtype(df, "age", np.int64)
        .check_no_nulls(df, "user_id")
        .check_unique(df, "user_id")
        .check_range(df, "age", 0, 120)
        .check_positive(df, "salary")
        .check_range(df, "score", 0, 1)
        .check_valid_values(df, "country", ["US", "CA", "UK", "MX"])
        .check_regex(df, "email", r"^[\w\.-]+@[\w\.-]+\.\w+$", "email format")
        .check_null_percentage(df, "email", max_percent=10.0)
        .validate()
    )
    return df


# USAGE IN PIPELINES
try:
    df_final = (
        df
        .pipe(validate_data)      # validate data at start
        # some operations
        .pipe(validate_data)      # validate again      
        # some operations
        .pipe(validate_data)      # validate data at end
    )
    print("✅ Pipeline completed successfully!")
except ValidationError as e:
    print(f"❌ Pipeline failed at validation step: {e}")
    print("Generating detailed validation report...")
    report = {
        "is_valid": False,
        "error_count": len(e.errors),
        "errors": e.errors,
    }
    print(f"  Total issues: {report['error_count']}")
    print(f"  Valid: {report['is_valid']}")
    if report['errors']:
        print("\n  Detailed issues:")
        for error in report['errors']:
            print(f"    - {error}")
