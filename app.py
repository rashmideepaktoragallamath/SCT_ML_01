from flask import Flask, render_template, request
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# Load Dataset
data = pd.read_csv("Housing (3).csv")

# Encode Categorical Columns
categorical = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

encoder = LabelEncoder()

for col in categorical:
    data[col] = encoder.fit_transform(data[col])

# Features and Target

X = data.drop("price", axis=1)
y = data["price"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# Model Evaluation

y_pred = model.predict(X_test)

r2 = round(
    r2_score(
        y_test,
        y_pred
    ),
    3
)

mae = round(
    mean_absolute_error(
        y_test,
        y_pred
    ),
    2
)

# Home Page

@app.route("/")
def home():

    return render_template(
        "index.html",
        r2=r2,
        mae=mae
    )

# Prediction

@app.route("/predict", methods=["POST"])
def predict():

    values = []

    for col in X.columns:
        values.append(
            float(
                request.form[col]
            )
        )

    prediction = model.predict([values])

    return render_template(
        "index.html",
        prediction=round(
            prediction[0],
            2
        ),
        r2=r2,
        mae=mae
    )

# Analytics Page

@app.route("/analytics")
def analytics():

    return render_template(
        "analytics.html",
        r2=r2,
        mae=mae
    )

# About Page

@app.route("/about")
def about():

    return render_template(
        "about.html",
        r2=r2,
        mae=mae
    )

# Run App

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)