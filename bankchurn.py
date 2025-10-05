
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import io # Import the io module

def preprocess_data(df, scaler):
    """
    Preprocesses the input DataFrame for prediction.

    Args:
        df: pandas DataFrame with raw data.
        scaler: Fitted StandardScaler object.

    Returns:
        pandas DataFrame with preprocessed data.
    """
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure all columns present during training are present in the new data
    # Add missing columns with default value 0
    training_columns = joblib.load('training_columns.pkl') # Assuming you saved training columns
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    # Reorder columns to match training data
    df_encoded = df_encoded[training_columns]


    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])

    return df_encoded

def predict_churn(data_source, model_path='gradient_boosting_model.pkl', scaler_path='scaler.pkl'):
    """
    Loads the model and scaler, preprocesses the data, and makes predictions.

    Args:
        data_source: Path to the new data file (e.g., CSV or Excel) or a file-like object.
        model_path: Path to the saved model file.
        scaler_path: Path to the saved scaler file.

    Returns:
        pandas Series with predictions.
    """
    # Load the model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load the data
    if isinstance(data_source, str):
        # Assume it's a file path
        if data_source.endswith('.xlsx'):
            df = pd.read_excel(data_source)
        elif data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
        else:
            raise ValueError("Unsupported file format for the given path. Please provide an Excel or CSV file.")
    elif isinstance(data_source, io.IOBase):
        # Assume it's a file-like object from Streamlit upload
        try:
            # Try reading as excel first
            df = pd.read_excel(data_source)
        except Exception as e_excel:
            # If excel reading fails, try reading as csv
            try:
                 # Reset the file pointer to the beginning before trying to read as CSV
                data_source.seek(0)
                df = pd.read_csv(data_source)
            except Exception as e_csv:
                 raise ValueError(f"Could not read data from the provided source. Tried Excel and CSV formats. Excel error: {e_excel}, CSV error: {e_csv}") from e_csv
    else:
        raise ValueError("Unsupported data source type. Please provide a file path (string) or a file-like object.")


    # Preprocess the data
    processed_df = preprocess_data(df, scaler)

    # Make predictions
    predictions = model.predict(processed_df)

    return predictions

# Remove the example usage block as it's not suitable for Streamlit deployment.
# if __name__ == '__main__':
#     # Example usage:
#     # Assuming you have a new data file named 'new_bank_data.xlsx'
#     # Replace with the actual path to your new data
#     new_data_path = '/content/bank-full.xlsx' # Replace with your new data path

#     # Make predictions
#     predictions = predict_churn(new_data_path)

#     # Display predictions
#     print("Predictions:")
#     print(predictions)
