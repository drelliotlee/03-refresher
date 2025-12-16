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
    
    def check_range(self, df: pd.DataFrame, col: str, min_val: float, max_val: float) -> 'DataValidator':
        """Check if column values are within range"""
        invalid = ~df[col].between(min_val, max_val)
        if invalid.any():
            count = invalid.sum()
            examples = df.loc[invalid, col].head(3).tolist()
            self.errors.append(
                f"{col}: {count} values outside range [{min_val}, {max_val}]. Examples: {examples}"
            )
        return self
    
    def check_no_nulls(self, df: pd.DataFrame, col: str) -> 'DataValidator':
        """Check for null values"""
        if df[col].isna().any():
            count = df[col].isna().sum()
            self.errors.append(f"{col}: {count} null values found")
        return self
    
    def check_unique(self, df: pd.DataFrame, col: str) -> 'DataValidator':
        """Check for duplicate values"""
        if df[col].duplicated().any():
            count = df[col].duplicated().sum()
            self.errors.append(f"{col}: {count} duplicate values found")
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
        """Check if all values are positive"""
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
def validate_user_data(df: pd.DataFrame) -> pd.DataFrame:
    validator = DataValidator()   
    (validator
        .check_no_nulls(df, "user_id")
        .check_unique(df, "user_id")
        .check_range(df, "age", 0, 120)
        .check_positive(df, "salary")
        .check_range(df, "score", 0, 1)
        .check_valid_values(df, "country", ["US", "CA", "UK", "MX"])
        .check_regex(df, "email", r"^[\w\.-]+@[\w\.-]+\.\w+$", "email format")
        .validate()
    )
    return df

def get_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    validator = DataValidator()
    (validator
        .check_range(df, "age", 0, 120)
        .check_positive(df, "salary")
        .check_range(df, "score", 0, 1)
        .check_valid_values(df, "country", ["US", "CA", "UK", "MX"])
    )
    return {
        "is_valid": len(validator.errors) == 0,
        "error_count": len(validator.errors),
        "errors": validator.errors,
    }


# USAGE IN PIPELINES
try:
    df_final = (
        df
        .pipe(validate_user_data)      # validate data at start
        # some operations
        .pipe(validate_user_data)      # validate again      
        # some operations
        .pipe(validate_user_data)      # validate data at end
    )
    print("✅ Pipeline completed successfully!")
except ValidationError as e:
    print(f"❌ Pipeline failed at validation step: {e}")
    print("Generating detailed validation report...")
    report = get_validation_report(df)
    print(f"  Total issues: {report['error_count']}")
    print(f"  Valid: {report['is_valid']}")
    if report['errors']:
        print("\n  Detailed issues:")
        for error in report['errors']:
            print(f"    - {error}")
