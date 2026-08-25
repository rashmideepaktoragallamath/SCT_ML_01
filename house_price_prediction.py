import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# --------------------------
# LOAD DATASET
# --------------------------

data = pd.read_csv("Housing (3).csv")

print("\nDataset Shape:")
print(data.shape)


# --------------------------
# FEATURES AND TARGET
# --------------------------

X = data.drop("price", axis=1)
y = data["price"]


# --------------------------
# COLUMN TYPES
# --------------------------

categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]


# --------------------------
# PREPROCESSING
# --------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# --------------------------
# BEST MODEL
# --------------------------

regressor = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# --------------------------
# COMPLETE PIPELINE
# --------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ]
)


# --------------------------
# TRAIN TEST SPLIT
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------
# TRAIN MODEL
# --------------------------

print("\nTraining final PropVision AI model...")

model.fit(X_train, y_train)


# --------------------------
# TEST MODEL
# --------------------------

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


# --------------------------
# DISPLAY PERFORMANCE
# --------------------------

print("\nFinal PropVision AI Model")
print("=" * 50)

print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")


# --------------------------
# SAVE MODEL
# --------------------------

model_path = "model/propvision_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(f"Location: {model_path}")

print("\nPhase 1 - Step 5 Completed Successfully!")