import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from itertools import product
from lifelines.utils import concordance_index

# ---------- Load training data ----------
# Data structure:
# Column 1: ID
# Column 2: event status (0 = censored, 1 = event)
# Column 3: survival time
# Column 4 onward: model features
train_data = pd.read_csv("train.csv")

# Prepare survival labels and model features
y_train = (
    np.where(train_data.iloc[:, 1] == 0, -1, 1)
    * train_data.iloc[:, 2]
)

x_train = train_data.iloc[:, 3:]

# Convert training data to XGBoost DMatrix
dtrain = xgb.DMatrix(x_train, label=y_train)

# ---------- Grid search ----------
best_score = 0
seed = seed # e.g. 100

for n_estimators, max_depth, min_child_weight, subsample, colsample in product(
 [],   # e.g. [5, 10, 20]
    [],   # e.g. [2, 3, 4]
    [],   # e.g. [1, 2, 3]
    [],   # e.g. [0.6, 0.8, 1.0]
    []    # e.g. [0.6, 0.8, 1.0]

):
    params = {
        "objective": "survival:cox",
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "seed": seed
    }

    # Train XGBoost Cox model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators
    )

    # Calculate Harrell's C-index
    score = concordance_index(
        train_data.iloc[:, 2],
        -model.predict(dtrain),
        train_data.iloc[:, 1]
    )

    # Update the best-performing model
    if score > best_score:
        best_score = score
        best_model = model
        best_params = {
            **params,
            "n_estimators": n_estimators
        }

# ---------- Report and save model ----------
print("Best C-index:", round(best_score, 3))
print("Best parameters:", best_params)

joblib.dump(best_model, "model_Xgboost.pkl")

# ---------- Load validation cohort ----------
# The validation dataset should have the same feature columns as the training data
validation_data = pd.read_csv("validation.csv")

x_validation = validation_data.iloc[:, 3:]
dvalidation = xgb.DMatrix(x_validation)

# ---------- Load trained model ----------
model = joblib.load("model_Xgboost.pkl")

# ---------- Generate prediction scores ----------
validation_data["prediction"] = model.predict(dvalidation)

# ---------- Save prediction results ----------
validation_data.to_csv(
    "validation_prediction.csv",
    index=False
)


