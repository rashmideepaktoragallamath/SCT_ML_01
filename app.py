import streamlit as st
import pandas as pd
import joblib


# --------------------------
# PAGE CONFIGURATION
# --------------------------

st.set_page_config(
    page_title="PropVision AI",
    page_icon="🏠",
    layout="wide"
)


# --------------------------
# LOAD MODEL
# --------------------------

model = joblib.load("model/propvision_model.pkl")


# --------------------------
# TITLE
# --------------------------

st.title("🏠 PropVision AI")

st.subheader(
    "AI-Powered Property Valuation & Decision Support"
)

st.write(
    "Enter the property details below to estimate its market value."
)


# --------------------------
# PROPERTY DETAILS
# --------------------------

st.header("Property Details")


col1, col2, col3 = st.columns(3)


with col1:

    area = st.number_input(
        "Area (sq ft)",
        min_value=300,
        max_value=50000,
        value=5000,
        step=100
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=10,
        value=2
    )


with col2:

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        value=2
    )

    mainroad = st.selectbox(
        "Main Road",
        ["yes", "no"]
    )

    guestroom = st.selectbox(
        "Guest Room",
        ["yes", "no"]
    )

    basement = st.selectbox(
        "Basement",
        ["yes", "no"]
    )


with col3:

    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["yes", "no"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["yes", "no"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["yes", "no"]
    )

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]
    )


# --------------------------
# PREDICTION BUTTON
# --------------------------

st.divider()

if st.button(
    "🔮 Predict Property Value",
    use_container_width=True
):

    property_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])


    predicted_price = model.predict(
        property_data
    )[0]


    # --------------------------
    # DISPLAY RESULT
    # --------------------------

    st.success("Property valuation generated successfully!")

    st.metric(
        "Estimated Property Value",
        f"₹{predicted_price:,.0f}"
    )


    st.info(
        "This estimate is generated using the trained "
        "PropVision AI Gradient Boosting model."
    )