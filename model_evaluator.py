import matplotlib.pyplot as plt
import numpy as np

class ModelEvaluator:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def evaluate(self):
        print("Evaluating model...")
        for ticker in self.data['Ticker'].unique():
            data_ticker = self.data[self.data['Ticker'] == ticker]
            sharpe_ratio = self._calculate_sharpe_ratio(data_ticker)
            max_drawdown = self._calculate_max_drawdown(data_ticker)
            print("Evaluation Results:")
            print(f"Performance for {ticker}:")
            print(f"  Sharpe Ratio: {sharpe_ratio}")
            print(f"  Maximum Drawdown: {max_drawdown}")
            print("-" * 20)

    def _calculate_sharpe_ratio(self, data):
        returns = data['Returns']
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        return sharpe_ratio

    def _calculate_max_drawdown(self, data):
        cumulative_returns = (1 + data['Returns']).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def plot_results(self, ticker):
        data_ticker = self.data[self.data['Ticker'] == ticker]
        if data_ticker.empty:
            print(f"No data available for ticker {ticker}")
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(data_ticker['Date'], data_ticker['MA20'], label='20-day MA')
            plt.plot(data_ticker['Date'], data_ticker['Close'], label='Close Price')
            plt.title(f'20-Day Moving Average and Close Price for {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
