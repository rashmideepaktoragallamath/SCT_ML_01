import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------
# LOAD DATASET
# --------------------------

data = pd.read_csv("Housing (3).csv")

print("\nFirst 5 Rows:\n")
print(data.head())

print("\nDataset Info:\n")
print(data.info())

# --------------------------
# ENCODE CATEGORICAL COLUMNS
# --------------------------

encoder = LabelEncoder()

categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# --------------------------
# FEATURES AND TARGET
# --------------------------

X = data.drop("price", axis=1)
y = data["price"]

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

model = LinearRegression()

model.fit(X_train, y_train)

# --------------------------
# PREDICTIONS
# --------------------------

y_pred = model.predict(X_test)

# --------------------------
# MODEL PERFORMANCE
# --------------------------

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

# --------------------------
# GRAPH 1
# AREA VS PRICE
# --------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    data["area"],
    data["price"]
)

plt.xlabel("Area")

plt.ylabel("Price")

plt.title("Area vs Price")

plt.savefig("static/graphs/area_price.png")
plt.close()

# --------------------------
# GRAPH 2
# BEDROOMS VS PRICE
# --------------------------

plt.figure(figsize=(8,5))

bed_price = data.groupby("bedrooms")["price"].mean()

bed_price.plot(kind="bar")

plt.title("Average Price by Bedrooms")

plt.xlabel("Bedrooms")

plt.ylabel("Average Price")

plt.savefig("static/graphs/area_price.png")
plt.close()

# --------------------------
# GRAPH 3
# BATHROOMS VS PRICE
# --------------------------

plt.figure(figsize=(8,5))

bath_price = data.groupby("bathrooms")["price"].mean()

bath_price.plot(kind="bar")

plt.title("Average Price by Bathrooms")

plt.xlabel("Bathrooms")

plt.ylabel("Average Price")

plt.savefig("static/graphs/area_price.png")
plt.close()
# --------------------------
# GRAPH 4
# PARKING VS PRICE
# --------------------------

plt.figure(figsize=(8,5))

parking_price = data.groupby("parking")["price"].mean()

parking_price.plot(kind="bar")

plt.title("Parking vs Average Price")

plt.xlabel("Parking")

plt.ylabel("Average Price")

plt.savefig("static/graphs/area_price.png")
plt.close()

# --------------------------
# GRAPH 5
# ACTUAL VS PREDICTED
# --------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.savefig("static/graphs/area_price.png")
plt.close()

print("\nProject Step 1 Completed Successfully!")