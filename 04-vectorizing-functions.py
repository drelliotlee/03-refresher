# the purpose of this refresher file is to
# show what i learned about vectorizing functions in pandas / numpy
# and when to use which method
#
# TLDR SUMMARY
#
# apply = bad
# np.vectorize = bad
# njit (for loops) = good
# numpy (for arithmetic) = good
# everything else = cant be sped up


# ------------------------------------------------------------
# Sample dataframe
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from numba import njit

n = 10_000
df = pd.DataFrame(
    {
        "colA": np.random.rand(n),
        "colB": np.random.rand(n),
        "some_col": np.random.rand(n),
        "text": np.random.choice(
            ["this is great", "bad example", "absolutely amazing", "terrible"],
            size=n,
        ),
    }
)

# ============================================================
# PART 1: arithmetic functions can be vectorized
# ============================================================

def arbitrary_func1(x, y):
    return x + 2 * y


# ---- BAD: pandas apply (Python per-row calls) ----
def add_feature_apply(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        f1=df.apply(lambda r: arbitrary_func1(r["colA"], r["colB"]), axis=1)
    )


# ---- BAD: np.vectorize (still Python per-element) ----
vectorized_bad = np.vectorize(arbitrary_func1)

def add_feature_np_vectorize(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        f1=vectorized_bad(df["colA"].to_numpy(), df["colB"].to_numpy())
    )


# ---- GOOD: true vectorization (NumPy arithmetic) ----
def vectorized_func(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(f1=df["colA"] + 2 * df["colB"])

# ============================================================
# PART 2: loop-based function (running sum)
# arbitrary_func2 requires state across rows
# ============================================================

def running_sum_python(arr):
    out = []
    total = 0.0
    for x in arr:
        total += x
        out.append(total)
    return np.array(out)


# ---- BAD: np.vectorize (still Python loop) ----
vectorized_running_sum = np.vectorize(lambda x, acc=[0.0]: acc.append(acc[-1] + x) or acc[-1])

def add_running_sum_np_vectorize(df: pd.DataFrame) -> pd.DataFrame:
    # this is intentionally awful
    arr = df["some_col"].to_numpy()
    result = vectorized_running_sum(arr)
    return df.assign(f2=result)


# ---- GOOD: numba njit (compiled loop) ----
@njit
def running_sum_numba(arr):
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        total += arr[i]
        out[i] = total
    return out


def vectorized_func(df: pd.DataFrame) -> pd.DataFrame:
    arr = df["some_col"].to_numpy(dtype=np.float64)
    return df.assign(f2=running_sum_numba(arr))


# ============================================================
# PART 3: function that cannot be sped up
# string + Python objects + dynamic logic
# ============================================================

POSITIVE = {"great", "amazing"}
NEGATIVE = {"bad", "terrible"}


def arbitrary_func3(s: str) -> int:
    score = 0
    for token in s.lower().split():
        if token in POSITIVE:
            score += 1
        elif token in NEGATIVE:
            score -= 1
    return score


# ---- Only possible implementation: Python apply ----
def vectorized_func(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        f3=df["text"].apply(arbitrary_func3)
    )

# ============================================================
# USAGE FOR ALL 3
# ============================================================

df_final = (
    df
    .pipe(vectorized_func)
)

