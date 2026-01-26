from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from typing import Callable
from functools import wraps
import time

import numpy as np
import pandas as pd


# ======================================================
# Dummy data for testing
# ======================================================

np.random.seed(42)

df = pd.DataFrame({
    'id': range(1, 101),
    'colA': np.random.choice([1, 2, 3, np.nan], 100),
    'colB': np.random.choice(['X', 'Y', 'Z', np.nan], 100),
    'value': np.random.randn(100) * 100,
    'category_id': np.random.choice([1, 2, 3, 4, 5], 100)  # Will become category explosion
})

other_df = pd.DataFrame({
    'id': range(1, 121),
    'extra_col': np.random.choice(['A', 'B', 'C'], 120),
    'score': np.random.randint(0, 100, 120)
})

# Introduce some duplicates
df = pd.concat([df, df.sample(5)], ignore_index=True)


# ======================================================
# Logging setup
# ======================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(
    LOG_DIR / "pipeline.log",
    maxBytes=5_000_000,
    backupCount=5,
)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Global tracker for pipeline metrics
_pipeline_metrics = []
_pipeline_warnings = []
_pipeline_errors = []


# ======================================================
# Observability decorator
# ======================================================

def observe_step(fn: Callable[[pd.DataFrame], pd.DataFrame]):
    @wraps(fn)
    def wrapper(df: pd.DataFrame) -> pd.DataFrame:
        step_name = fn.__name__
        start_time = time.time()
        
        # Track all BEFORE metrics
        rows_before = len(df)
        cols_before = set(df.columns)
        dtypes_before = df.dtypes.to_dict()
        numeric_before = df.select_dtypes(include="number")
        stats_before = {}
        if not numeric_before.empty:
            stats_before = numeric_before.agg(["min", "max", "mean"]).to_dict()
        
        # Execute the Pipeline
        try:
            out = fn(df)
        except Exception as e:
            _pipeline_errors.append(f"{step_name} | FAILED: {type(e).__name__}: {str(e)}")
            logger.error(f"{step_name} | FAILED: {type(e).__name__}: {str(e)}")
            raise
        
        execution_time = time.time() - start_time
        rows_after = len(out)
        cols_after = set(out.columns)
        delta = rows_after - rows_before
        cols_changed = len(cols_before.symmetric_difference(cols_after))

        # Collect metrics for summary table
        _pipeline_metrics.append({
            'step': step_name,
            'delta_rows': delta,
            'cols_added_removed': cols_changed,
            'execution_time': execution_time
        })

        # 4. Warn when column dtypes changed (helps debug which step broke schema)
        dtypes_after = out.dtypes.to_dict()
        dtype_changes = {}
        for col in set(dtypes_before.keys()) & set(dtypes_after.keys()):
            if dtypes_before[col] != dtypes_after[col]:
                dtype_changes[col] = f"{dtypes_before[col]} -> {dtypes_after[col]}"
        if dtype_changes:
            _pipeline_warnings.append(
                f"{step_name} | dtype_changes={dtype_changes}"
            )
        
        # 5. Warn on significant numeric changes (>30% relative change) - DATA DRIFT
        numeric_after = out.select_dtypes(include="number")
        stats_after = {}
        if not numeric_after.empty:
            stats_after = numeric_after.agg(["min", "max", "mean"]).to_dict()
            
            significant_changes = {}
            for col in set(stats_before.keys()) & set(stats_after.keys()):
                for stat in ["min", "max", "mean"]:
                    before_val = stats_before[col][stat]
                    after_val = stats_after[col][stat]
                    if before_val != 0:
                        pct_change = abs((after_val - before_val) / before_val)
                        if pct_change > 0.3:
                            significant_changes[f"{col}_{stat}"] = f"{before_val:.2f} -> {after_val:.2f} ({pct_change*100:.1f}%)"
            
            if significant_changes:
                _pipeline_warnings.append(
                    f"{step_name} | significant_numeric_changes={significant_changes}"
                )

        return out
    return wrapper

# ======================================================
# Pipeline steps
# ======================================================

@observe_step
def merge_other_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(other_df, how='left', on='id')

@observe_step
def accidentally_convert_to_string(df: pd.DataFrame) -> pd.DataFrame:
    # Common mistake: numeric ID accidentally converted to string (dtype change)
    df = df.copy()
    df['id'] = df['id'].astype(str)
    return df

@observe_step
def accidentally_explode_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Common mistake: creating high cardinality by concatenating fields
    df = df.copy()
    df['category_id'] = df['category_id'].astype(str) + '_' + df['extra_col'] + '_' + df.index.astype(str)
    return df

@observe_step
def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['colA', 'colB'])

@observe_step
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=['id'])

# ======================================================
# Pipeline execution
# ======================================================

df_final = (
    df
    .pipe(merge_other_df)
    .pipe(accidentally_convert_to_string)
    .pipe(accidentally_explode_categories)
    .pipe(drop_missing_rows)
    .pipe(remove_duplicates)
    )

# Log summary table
summary_df = pd.DataFrame(_pipeline_metrics)
sep = "=" * 60
output = f"\n{sep}\nPIPELINE SUMMARY\n{sep}\n{summary_df.to_string(index=False)}\n{sep}"

if _pipeline_warnings:
    output += "\nWARNINGS:\n"
    for warning in _pipeline_warnings:
        output += f"  - {warning}\n"
    output += sep

logger.info(output)