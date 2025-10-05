
import streamlit as st
import pandas as pd
from bankchurn import predict_churn # Import the predict_churn function

st.title("Bank Term Deposit Prediction")

st.write("Upload your bank data to predict term deposit subscription.")

uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # When a file is uploaded, call the predict_churn function
    predictions = predict_churn(uploaded_file)

    st.subheader("Predictions")

    # Display the predictions
    prediction_df = pd.DataFrame({'Prediction': predictions})
    st.write(prediction_df)

    st.success("Predictions generated successfully!")

else:
    st.info("Please upload a file to get predictions.")
