import streamlit as st
import pandas as pd
import joblib
import os

# Load files
base_path = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_path, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
training_columns = joblib.load(os.path.join(base_path, "training_columns.pkl"))

# Title
st.title("Online Shoppers Prediction")

# Description 
st.write("Enter user session details to predict whether the user will purchase or not.")

# Model info
st.info("Model used: Random Forest")

# Inputs
administrative = float(st.text_input("Administrative", "0"))
informational = float(st.text_input("Informational", "0"))
product_related = float(st.text_input("ProductRelated", "0"))
bounce_rates = float(st.text_input("BounceRates", "0.0"))
exit_rates = float(st.text_input("ExitRates", "0.0"))
page_values = float(st.text_input("PageValues", "0.0"))

weekend = st.selectbox("Weekend", [False, True])
visitor_type = st.selectbox("VisitorType", ["Returning_Visitor", "New_Visitor", "Other"])
month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

# Prediction
if st.button("Predict"):
    
    input_data = pd.DataFrame({
        "Administrative": [administrative],
        "Informational": [informational],
        "ProductRelated": [product_related],
        "BounceRates": [bounce_rates],
        "ExitRates": [exit_rates],
        "PageValues": [page_values],
        "Weekend": [weekend],
        "VisitorType": [visitor_type],
        "Month": [month]
    })

    # Encoding
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=training_columns, fill_value=0)

    # Show input 
    st.subheader("Input Data:")
    st.write(input_data)

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Result 
    if prediction == 1:
        st.success("✅ User WILL purchase")
    else:
        st.error("❌ User will NOT purchase")
