import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

# Download stock data using yfinance
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2015-01-01', end='2023-01-01')
data['Close'].plot(title=f'{stock_symbol} Stock Prices')
plt.show()

# Compute and plot Moving Average
data['Moving_Average'] = data['Close'].rolling(window=30).mean()
data[['Close', 'Moving_Average']].plot(title=f'{stock_symbol} Moving Average')
plt.show()

# Apply Exponential Smoothing
model_es = ExponentialSmoothing(data['Close'], seasonal='mul', seasonal_periods=12).fit()
data['Exponential_Smoothing'] = model_es.fittedvalues
data[['Close', 'Exponential_Smoothing']].plot(title=f'{stock_symbol} Exponential Smoothing')
plt.show()

# Fit ARIMA Model
model_arima = ARIMA(data['Close'], order=(5, 1, 0))
model_fit = model_arima.fit()
data['ARIMA'] = model_fit.fittedvalues
data[['Close', 'ARIMA']].plot(title=f'{stock_symbol} ARIMA')
plt.show()

# Forecast using ARIMA model
forecast = model_fit.forecast(steps=30)
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Close'], label='Historical')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.title(f'{stock_symbol} Stock Price Forecast')
plt.show()

# Split data into training and testing
train_size = int(len(data) * 0.8)
train, test = data['Close'][0:train_size], data['Close'][train_size:]

# Fit ARIMA on training set
model_arima_train = ARIMA(train, order=(5, 1, 0)).fit()
predictions = model_arima_train.forecast(steps=len(test))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'RMSE: {rmse}')
