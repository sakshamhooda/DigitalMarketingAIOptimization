from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class TimeSeriesModels:
    @staticmethod
    def sarima_forecast(data, steps=30):
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit()
        return results.forecast(steps=steps)

    @staticmethod
    def prophet_forecast(data, periods=30):
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=periods)
        return model.predict(future)

    @staticmethod
    def lstm_forecast(X_train, y_train, X_test, n_steps, n_features):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, batch_size=32)
        return model.predict(X_test)