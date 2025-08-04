import ccxt
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization
import altair as alt
from datetime import datetime, timedelta
import os

class TokenSurgeTracker:
    def __init__(self, symbol='ADA/USDT', timeframe='1h', lookback_days=12, api_key=None, api_secret=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = ccxt.kucoin({'apiKey': api_key, 'secret': api_secret})
        self.scaler = MinMaxScaler()

    def fetch_ohlcv(self):
        """Получение исторических данных с KuCoin."""
        since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_social_metrics(self):
        """Получение социальных метрик с X (заглушка для API X)."""
        # Реальная версия требует API X для анализа метрик
        return np.random.rand(len(self.fetch_ohlcv())) * 60

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Расчет Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def calculate_stochastic_oscillator(self, df, period=14):
        """Расчет Stochastic Oscillator."""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        stoch = 100 * (df['close'] - low_min) / (high_max - low_min)
        return stoch

    def prepare_data(self, df):
        """Подготовка данных для модели."""
        df['returns'] = df['close'].pct_change()
        df['upper_bb'], df['lower_bb'] = self.calculate_bollinger_bands(df['close'])
        df['stochastic'] = self.calculate_stochastic_oscillator(df)
        df['social_metrics'] = self.fetch_social_metrics()
        features = df[['close', 'stochastic', 'upper_bb', 'social_metrics']].dropna()

        scaled_data = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(1 if scaled_data[i, 0] > np.percentile(scaled_data[:, 0], 88) else 0)  # Ценовой импульс
        return np.array(X), np.array(y)

    def build_model(self):
        """Создание Transformer модели."""
        inputs = Sequential([
            MultiHeadAttention(num_heads=4, key_dim=4, input_shape=(60, 4)),
            Dropout(0.2),
            LayerNormalization(epsilon=1e-6),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model = Sequential([inputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y):
        """Обучение модели."""
        model = self.build_model()
        model.fit(X, y, epochs=12, batch_size=16, validation_split=0.2, verbose=1)
        return model

    def predict_surge(self, model, X):
        """Прогноз ценовых импульсов."""
        predictions = model.predict(X)
        return (predictions > 0.5).astype(int)

    def visualize_results(self, df, predictions):
        """Визуализация с Altair."""
        df = df.iloc[60:].copy()
        df['surge_prediction'] = predictions

        chart = alt.Chart(df).mark_line().encode(
            x='timestamp:T',
            y='close:Q',
            color=alt.value('blue'),
            tooltip=['timestamp', 'close']
        ).properties(title=f'Price Surges for {self.symbol}')

        points = alt.Chart(df[df['surge_prediction'] == 1]).mark_circle(size=100).encode(
            x='timestamp:T',
            y='close:Q',
            color=alt.value('red'),
            tooltip=['timestamp', 'close']
        )

        combined = (chart + points).interactive()
        combined.save('data/sample_output/surge_chart.html')

    def run(self):
        """Основной метод анализа."""
        df = self.fetch_ohlcv()
        X, y = self.prepare_data(df)
        model = self.train_model(X, y)
        predictions = self.predict_surge(model, X)
        self.visualize_results(df, predictions)
        print(f"Token surges predicted: {np.sum(predictions)} out of {len(predictions)} periods.")

if __name__ == "__main__":
    tracker = TokenSurgeTracker(symbol='ADA/USDT', timeframe='1h', lookback_days=12)
    tracker.run()
