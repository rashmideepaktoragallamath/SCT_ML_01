import pandas as pd
import joblib


# --------------------------
# LOAD SAVED MODEL
# --------------------------

model = joblib.load("model/propvision_model.pkl")

print("PropVision AI model loaded successfully!")


# --------------------------
# NEW PROPERTY
# --------------------------

property_data = pd.DataFrame([{
    "area": 5000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "yes",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "semi-furnished"
}])


# --------------------------
# PREDICT PRICE
# --------------------------

predicted_price = model.predict(property_data)[0]


# --------------------------
# DISPLAY RESULT
# --------------------------

print("\nProperty Details:")
print(property_data.to_string(index=False))

print("\nPredicted Property Price:")
print(f"₹{predicted_price:,.2f}")


print("\nPhase 1 - Step 6 Completed Successfully!")