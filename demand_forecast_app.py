
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(layout="wide")
st.title('Demand Forecasting with Time Series Models')

# --- 1. Data Loading --- #
st.header('1. Data Loading and Preprocessing')

@st.cache_data
def load_data(file_path='demand.csv'):
    data = pd.read_csv(file_path)
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    return data

df = load_data()

st.subheader('Raw Data Preview')
st.write(df.head())

# --- 2. Data Preprocessing --- #

st.subheader('Handling Missing Values')
# Fill the missing value in 'total_price' with the median of 'total_price' for the corresponding 'sku_id'
median_price_per_sku = df.groupby('sku_id')['total_price'].median()
df['total_price'].fillna(df['sku_id'].map(median_price_per_sku), inplace=True)
st.write(f"Missing values after filling: {df['total_price'].isnull().sum()}")


st.subheader('Aggregating Data to Weekly Frequency')
df.set_index('week', inplace=True)
weekly_data = df['units_sold'].resample('W').sum()
st.write(weekly_data.head())

# --- 3. Model Training and Prediction --- #
st.header('2. Model Training and Evaluation')

# Split data into train and test sets
train_data = weekly_data[:int(0.8 * len(weekly_data))]
test_data = weekly_data[int(0.8 * len(weekly_data)):]

# Initialize RMSE storage
rmse_results = {}
predictions = {}

# --- ARIMA Model ---
st.subheader('ARIMA Model')
try:
    with st.spinner('Training ARIMA Model...'):
        arima_model = ARIMA(train_data, order=(1, 0, 0)).fit()
        arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
        arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))
        rmse_results['ARIMA'] = arima_rmse
        predictions['ARIMA'] = arima_predictions
    st.success(f'ARIMA RMSE: {arima_rmse:.2f}')
except Exception as e:
    st.error(f'ARIMA Model training failed: {e}')

# --- Prophet Model ---
st.subheader('Prophet Model')
try:
    # Prepare data for Prophet
    prophet_data = weekly_data.reset_index()
    prophet_data.columns = ['ds', 'y']
    train_data_prophet = prophet_data[:int(0.8 * len(prophet_data))]
    test_data_prophet = prophet_data[int(0.8 * len(prophet_data)):]

    with st.spinner('Training Prophet Model...'):
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        prophet_model.fit(train_data_prophet)

        # Create future dataframe for predictions
        prophet_future = prophet_model.make_future_dataframe(periods=len(test_data), freq='W')
        prophet_predictions_df = prophet_model.predict(prophet_future)
        
        # Extract predictions for the test period
        prophet_predictions = prophet_predictions_df.set_index('ds')['yhat'].loc[test_data.index]
        prophet_rmse = sqrt(mean_squared_error(test_data, prophet_predictions))
        rmse_results['Prophet'] = prophet_rmse
        predictions['Prophet'] = prophet_predictions
    st.success(f'Prophet RMSE: {prophet_rmse:.2f}')
except Exception as e:
    st.error(f'Prophet Model training failed: {e}')


# --- 4. Model Comparison --- #
st.header('3. Model Comparison')
if rmse_results:
    best_model_name = min(rmse_results, key=rmse_results.get)
    st.write(f"The best performing model is: **{best_model_name}** with RMSE: {rmse_results[best_model_name]:.2f}")
    st.write("RMSEs for all models:", rmse_results)
else:
    st.warning("No models were successfully trained.")

# --- 5. Visualization --- #
st.header('4. Forecast Visualization')

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(weekly_data.index, weekly_data, label='Original Data')
ax.plot(train_data.index, train_data, label='Training Data', color='blue')
ax.plot(test_data.index, test_data, label='Actual Test Data', color='orange')

for model_name, model_predictions in predictions.items():
    ax.plot(test_data.index, model_predictions, label=f'{model_name} Forecast', linestyle='--')

ax.set_title('Demand Forecast Comparison')
ax.set_xlabel('Week')
ax.set_ylabel('Units Sold')
ax.legend()
ax.grid(True)
st.pyplot(fig)