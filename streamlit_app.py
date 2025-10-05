
import streamlit as st
import pandas as pd
import joblib
from bankchurn import preprocess_data # Import preprocess_data

st.title("Bank Term Deposit Prediction")

st.write("Enter the customer details below to predict term deposit subscription.")

# Load the scaler and training columns
try:
    scaler = joblib.load('scaler.pkl')
    training_columns = joblib.load('training_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'scaler.pkl' and 'training_columns.pkl' are in the same directory.")
    st.stop()

# Create input fields for each feature
# You'll need to customize these based on your dataset's features and their types
st.subheader("Customer Details")

# Example input fields (replace with your actual features)
# You'll need to determine the appropriate Streamlit widget for each feature (st.number_input, st.text_input, st.selectbox, etc.)
# based on the data type and range of the feature.

age = st.number_input("Age", min_value=18, max_value=100, value=40)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Has Credit Default?", ['no', 'yes'])
balance = st.number_input("Balance", value=1000)
housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
contact = st.selectbox("Contact Communication Type", ['unknown', 'cellular', 'telephone'])
day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
month = st.selectbox("Last Contact Month of Year", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=200)
campaign = st.number_input("Number of Contacts During This Campaign", min_value=1, value=2)
pdays = st.number_input("Days Since Last Contact from Previous Campaign", min_value=-1, value=-1)
previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, value=0)
poutcome = st.selectbox("Outcome of the Previous Marketing Campaign", ['unknown', 'other', 'failure', 'success'])


if st.button("Predict Churn"):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # Preprocess the input data
    processed_input_data = preprocess_data(input_data, scaler)

    # Load the model
    try:
        model = joblib.load('gradient_boosting_model.pkl')
    except FileNotFoundError:
        st.error("Gradient Boosting model file not found. Please ensure 'gradient_boosting_model.pkl' is in the same directory.")
        st.stop()

    # Make prediction
    prediction = model.predict(processed_input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("Prediction: The customer is likely to subscribe to a term deposit.")
    else:
        st.success("Prediction: The customer is unlikely to subscribe to a term deposit.")
