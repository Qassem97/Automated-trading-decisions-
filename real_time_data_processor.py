
import yfinance as yf


class RealTimeDataProcessor:
    def __init__(self):
        pass

    def add_features(self, df):
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std()
        df['Volume_norm'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean()

        # Replace deprecated fillna method with direct backfilling
        df['MA20'] = df['MA20'].bfill()
        df['MA50'] = df['MA50'].bfill()
        df['Returns'] = df['Returns'].fillna(0)  # Fill NaN returns with 0 as it indicates no change
        df['Volatility'] = df['Volatility'].bfill()
        df['Volume_norm'] = df['Volume_norm'].bfill()
        return df

    def fetch_and_process(self, ticker):
        # Fetch recent data for the ticker
        data = yf.download(ticker, period="1d", interval="1m")
        if data.empty:
            raise Exception("Failed to retrieve data")

        # Check if 'Date' column exists and handle it safely
        if 'Date' in data.columns:
            data.drop(columns=['Date'], inplace=True)  # Drop the 'Date' column if not needed for processing

        # Add necessary features
        data = self.add_features(data)
        return data.tail(1)  # Return only the most recent data point




