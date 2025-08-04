# TokenSurgeTracker

**TokenSurgeTracker** is a Python tool designed to forecast "token surges" — sudden, high-magnitude price movements in cryptocurrencies. It combines market data from KuCoin with technical indicators (Bollinger Bands, Stochastic Oscillator) and social media metrics (e.g., X activity) to predict surges using a Transformer-based neural network. The tool generates interactive visualizations with Altair, enabling traders to explore potential market opportunities.

## Features
- Fetches real-time OHLCV data from KuCoin.
- Integrates social media metrics for improved predictions.
- Uses a Transformer model to detect token surges.
- Generates interactive visualizations with Altair.
- Configurable for different symbols and timeframes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TokenSurgeTracker.git
   cd TokenSurgeTracker
