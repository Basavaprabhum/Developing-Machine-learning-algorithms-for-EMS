import numpy as np
import pandas as pd
import os
from keras.models import load_model
from matplotlib import pyplot as plt
import joblib
from datetime import datetime

# Load and prepare data
def load_and_prepare_data(file_path):
    file_name = r"data_set_hourly.xlsx"
    excel_file_path = os.path.join(file_path, file_name)
    df = pd.read_excel(excel_file_path)
    datetime_col = df['DateTime']
    features_to_scale = df.drop(['DateTime'], axis=1)
    scaler = joblib.load(r"C:\Users\basav\PycharmProjects\pythonProject1\finalCode\MPC loop\Models\scalar.pkl")
    scaled_features = scaler.transform(features_to_scale)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
    scaled_features_df['DateTime'] = datetime_col
    pv_power_index = scaled_features_df.columns.get_loc('PV Power(kWh)')
    return scaled_features_df, scaler, pv_power_index

# Load all models
def load_all_models(model_folder_path):
    model_files = [file for file in os.listdir(model_folder_path) if file.endswith('.h5')]
    models = [load_model(os.path.join(model_folder_path, file)) for file in model_files]

    return models

# Make ensemble predictions
def make_ensemble_predictions(models, input_data, scaler, n_past, n_future, pv_power_index, datetime_col):
    final_predictions_dict = {}
    for i in range(len(input_data) - n_past + 1):
        val_sequence = input_data.iloc[i:i + n_past, :]
        val_sequence = val_sequence.to_numpy().reshape((1, val_sequence.shape[0], val_sequence.shape[1]))
        avg_predicted_val = np.zeros((n_future,))
        for model in models:
            predicted_val = model.predict(val_sequence).flatten()
            predicted_val = predicted_val[:n_future]
            avg_predicted_val += predicted_val
        avg_predicted_val /= len(models)
        dummy_val_array = np.zeros((n_future, input_data.shape[1]))
        dummy_val_array[:, pv_power_index] = avg_predicted_val
        predicted_val_rescaled = scaler.inverse_transform(dummy_val_array)[:, pv_power_index]
        prediction_start_date = datetime_col.iloc[i + n_past - 1] + pd.Timedelta(hours=1)
        prediction_dates = pd.date_range(start=prediction_start_date, periods=n_future, freq='H')
        for date, pred in zip(prediction_dates, predicted_val_rescaled):
            final_predictions_dict[date] = pred
    return final_predictions_dict

# New function to select data for prediction
def select_data_for_prediction(preprocessed_data, start_datetime, end_datetime, current_iteration):
    if 'DateTime' not in preprocessed_data.columns:
        raise ValueError("The 'DateTime' column is missing in the dataset.")
    datetime_col = pd.to_datetime(preprocessed_data['DateTime'])
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    # Find the index of the start_datetime or the next closest timestamp in the data
    start_index = datetime_col.searchsorted(start_datetime, side='right') - 50
    # The end index should be the index of start_datetime plus the current iteration
    end_index = start_index + 50 + current_iteration

    start_index = max(start_index, 0)  # Ensure start_index is not negative
    end_index = min(end_index, len(datetime_col))  # Ensure end_index does not exceed the length of the data

    selected_data = preprocessed_data.iloc[start_index:end_index]
    return selected_data

# Class for PV prediction
class PVPrediction:
    def __init__(self, preprocessed_data, start_datetime, end_datetime, n_past, n_future, model_folder_path):
        self.preprocessed_data = preprocessed_data
        self.start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        self.end_datetime = datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')
        self.total_hours = int((self.end_datetime - self.start_datetime).total_seconds() / 3600)
        self.current_iteration = 0
        self.models = load_all_models(model_folder_path)
        self.scaler = scaler
        self.n_past = n_past
        self.n_future = n_future
        self.pv_power_index = pv_power_index

    def run_once(self, n_future,total_hours):

        if self.current_iteration <= self.total_hours:
            selected_data = select_data_for_prediction(self.preprocessed_data, self.start_datetime, self.end_datetime, self.current_iteration)
            datetime_col = selected_data['DateTime']
            print(datetime_col)
            # Drop the 'DateTime' column for ensemble predictions
            selected_data_features = selected_data.drop(['DateTime'], axis=1)
            predictions_dict = make_ensemble_predictions(self.models, selected_data_features, self.scaler, self.n_past, n_future, self.pv_power_index, datetime_col)
            predictions_df = pd.DataFrame(list(predictions_dict.items()), columns=['DateTime', 'Predicted PV Power(kWh)'])
            print(predictions_df)
            last_n_future_predictions = predictions_df['Predicted PV Power(kWh)'].tail(total_hours)
            last_n_future_predictions_clipped = last_n_future_predictions.clip(lower=0)

            modified_array = []
            for value in last_n_future_predictions_clipped:
                divided_values = [value / 2] * 2  # Create a list of four equal parts
                modified_array.extend(divided_values)  # Append to the new array

            print("Modified Array:", modified_array)

            self.current_iteration += 1
            return modified_array
        else:
            print("Prediction process is complete.")

# Load and preprocess data
file_path = r"C:\Users\basav\PycharmProjects\pythonProject1\finalCode\MPC loop"
preprocessed_data, scaler, pv_power_index = load_and_prepare_data(file_path)