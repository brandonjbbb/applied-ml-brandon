"""
ml01.py  —  California Housing Baseline Analysis
Author: Brandon

This notebook-style script establishes a transparent baseline for predicting California
median house values using linear regression. It is intentionally simple, emphasizing
reproducibility, interpretability, and defensible methodology over brute performance.

Context:
    - Uses the built-in California Housing dataset as a clean, low-friction regression benchmark.
    - Employs a limited feature subset to isolate signal strength and avoid confounding.
    - Introduces a constant-value baseline model for context on what “learning” adds.
    - Evaluates model fit with R², MAE, and RMSE; inspects residuals and coefficients.
"""

# ---------------------------------------------------------------------------
# 0. Imports (all required libraries pre-installed via uv sync)
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import cast, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# 1. Load Dataset and Verify Integrity
# ---------------------------------------------------------------------------
X_df, y_ser = fetch_california_housing(as_frame=True, return_X_y=True)
X: pd.DataFrame = cast(pd.DataFrame, X_df)
y: pd.Series = cast(pd.Series, y_ser)

df: pd.DataFrame = X.copy()
df["MedHouseVal"] = y

assert not df.isnull().any().any(), "Dataset unexpectedly contains missing values."
assert df.select_dtypes("number").shape[1] == df.shape[1], "Non-numeric columns found."

# quick visual distribution check — confirms dataset integrity
df.hist(bins=30, figsize=(12, 8))
plt.suptitle("Feature Distributions — California Housing", fontsize=12)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 2. Stratified Split (by Target Deciles)
# ---------------------------------------------------------------------------
# Stratifying by target preserves distribution balance between train/test sets.
qbins = pd.qcut(df["MedHouseVal"], q=10, duplicates="drop")
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, qbins))

train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

features: List[str] = ["MedInc", "AveRooms"]
target: str = "MedHouseVal"
X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# ---------------------------------------------------------------------------
# 3. Baseline vs. Linear Regression
# ---------------------------------------------------------------------------
baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)
y_base = baseline.predict(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ---------------------------------------------------------------------------
# 4. Metric Reporting
# ---------------------------------------------------------------------------
def report_metrics(name: str, y_true: pd.Series, y_hat: np.ndarray) -> dict:
    r2 = r2_score(y_true, y_hat)
    mae = mean_absolute_error(y_true, y_hat)
    rmse = root_mean_squared_error(y_true, y_hat)
    print(f"{name:>10} | R²={r2:0.3f} | MAE={mae:0.3f} | RMSE={rmse:0.3f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}


print("\nModel Evaluation (Lower MAE/RMSE = Better, Higher R² = Better)")
base_m = report_metrics("Baseline", y_test, y_base)
lin_m = report_metrics("LinearRG", y_test, y_pred)

impr = 100 * (base_m["mae"] - lin_m["mae"]) / base_m["mae"]
print(f"→ MAE improvement vs baseline: {impr:0.1f}%\n")

# ---------------------------------------------------------------------------
# 5. Residual Diagnostics
# ---------------------------------------------------------------------------
residuals = y_test - y_pred

plt.figure(figsize=(5, 4))
plt.scatter(y_pred, residuals, s=10, alpha=0.6)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (y - ŷ)")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
plt.hist(residuals, bins=30, color="steelblue", alpha=0.8)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 6. Coefficients Table (Interpretability)
# ---------------------------------------------------------------------------
coef_df = (
    pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    .assign(Intercept=model.intercept_)
    .sort_values("Coefficient", key=np.abs, ascending=False)
)
print("Coefficient Summary:")
print(coef_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Visual Validation — Actual vs Predicted
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
plt.scatter(test_df["MedInc"], y_test, s=8, alpha=0.4, label="Actual")
plt.scatter(test_df["MedInc"], y_pred, s=8, alpha=0.7, label="Predicted")
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("MedInc vs MedHouseVal — Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()
