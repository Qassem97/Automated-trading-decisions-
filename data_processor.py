import numpy as np
from sklearn.model_selection import train_test_split
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def add_features(self):
        # Original feature engineering steps
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        self.data['Volume_norm'] = self.data['Volume'] / self.data['Volume'].max()
        self.data.dropna(inplace=True)

    def process_data(self, features, target):
        # Ensuring data is clean and prepared for training
        self.add_features()  # Ensure features are added before splitting
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)
        assert not self.data.isnull().any().any(), "NaN values exist in the DataFrame after handling."

        X = self.data[features]
        y = self.data[target]

        # Splitting data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
