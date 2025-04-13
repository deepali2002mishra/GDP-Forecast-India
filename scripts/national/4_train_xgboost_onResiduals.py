import pandas as pd
import xgboost as xgb
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# === Load Data ===
df = pd.read_csv("data/processed/processed_data.csv")
sarimax_df = pd.read_csv("data/processed/sarimax_predictions.csv")

# Merge SARIMAX predictions into data
df = df.merge(sarimax_df[["Year", "SARIMAX_Pred"]], on="Year", how="left")

# Compute residuals
df["Residual"] = df["GDP Growth (%)"] - df["SARIMAX_Pred"]

# Drop rows with missing values
df.dropna(subset=["Residual"], inplace=True)

# Define features (excluding metadata and target columns)
exclude_cols = ["Year", "GDP Growth (%)", "SARIMAX_Pred", "Residual"]
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

X = df[feature_cols]
y = df["Residual"]

# Create output directory if not exists
os.makedirs("models", exist_ok=True)

# TimeSeriesSplit for validation
tscv = TimeSeriesSplit(n_splits=3)
rmse_scores = []

for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.025,
        "max_depth": 4,
        "lambda": 2.0,
        "alpha": 1.0,
        "eval_metric": "rmse"
    }

    # Train model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=350,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Evaluate on validation
    y_pred = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {i+1} RMSE: {rmse:.3f}")

# Average RMSE
print(f"\n✅ Unified Residual Model Avg RMSE: {np.mean(rmse_scores):.3f}")

# Save trained model
model.save_model("models/xgb_residual.json")
print("📦 Model saved to models/xgb_residual.json")
