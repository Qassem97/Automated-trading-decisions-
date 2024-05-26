import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from CNN import CNN


class DataProcessor:
    def __init__(self, df):
        #self.df = df.reset_index(drop=True)  # Ensure the index is reset and does not interfere
        self.df = df

    def add_features(self):
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
        self.df['Volume_norm'] = self.df['Volume'] / self.df['Volume'].rolling(window=20).mean()

        # Handling NaN values
        self.df['MA20'] = self.df['MA20'].bfill()
        self.df['MA50'] = self.df['MA50'].bfill()
        self.df['Returns'].fillna(0, inplace=True)
        self.df['Volatility'].fillna(0, inplace=True)
        self.df['Volume_norm'].fillna(1, inplace=True)

        # Check for remaining NaN values
        nan_counts_after_filling = self.df.isna().sum()
        if nan_counts_after_filling.any():
            print("NaN values remaining after filling:")
            print(nan_counts_after_filling)
            raise ValueError("NaN values found after adding features")

    def cnn_feature_extraction(self, sequence_length=30):
        self.add_features()

        features = ['Close', 'MA20', 'MA50', 'Returns', 'Volatility', 'Volume_norm']
        data_sequences = self.create_sequences(self.df[features].values, sequence_length)

        X = data_sequences.transpose(0, 2, 1)  # [num_samples, num_features, sequence_length]

        model = CNN(sequence_length, len(features))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        model.train()

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(torch.tensor(X, dtype=torch.float32))

            # Ensure the lengths match for loss calculation
            cnn_output_length = outputs.size(0)
            returns_target = self.df['Returns'].iloc[sequence_length:sequence_length + cnn_output_length].values

            # Adjusting length to ensure they match exactly
            returns_target = returns_target[:cnn_output_length]

            loss = criterion(outputs, torch.tensor(returns_target, dtype=torch.float32).unsqueeze(1))
            loss.backward()
            optimizer.step()

        cnn_features = outputs.detach().numpy()

        # Correct padding to match the DataFrame length
        padding_length = len(self.df) - len(cnn_features)
        padding = np.full((padding_length, cnn_features.shape[1]), np.nan)
        cnn_features_list = np.vstack([padding, cnn_features])

        assert len(cnn_features_list) == len(self.df), "Length mismatch between CNN features and DataFrame"

        self.df['CNN_Features'] = [list(cnn_features_list[i]) for i in range(len(cnn_features_list))]

        # Drop rows with NaNs in CNN_Features
        nan_indices = self.df.index[self.df['CNN_Features'].apply(lambda x: any(np.isnan(x)))].tolist()
        self.df.drop(index=nan_indices, inplace=True)

        print(f"Dropped rows with NaNs in CNN_Features: {nan_indices}")

        # Debugging: Check for NaN values in final CNN_Features column
        nan_check = self.df['CNN_Features'].apply(lambda x: any(np.isnan(x)))
        if nan_check.any():
            nan_indices = nan_check[nan_check].index.tolist()
            print(f"NaN values found in CNN_Features column at indices: {nan_indices}")
            raise ValueError(f"CNN_Features column contains NaN values after assignment at indices: {nan_indices}")

    def prepare_data_for_env(self):
        return self.df

    @staticmethod
    def create_sequences(data, seq_length):
        xs = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            xs.append(x)
        return np.array(xs)
