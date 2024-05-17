import pandas as pd
import os
from datetime import datetime

class DataLoader:
    def __init__(self, directory_path, processed_file_path):
        self.directory_path = directory_path
        self.processed_file_path = processed_file_path

    def load_data(self):
        # Check if the processed file exists
        if os.path.exists(self.processed_file_path):
            print(f"Loading data from {self.processed_file_path}")
            return pd.read_parquet(self.processed_file_path)
        else:
            return self._read_and_combine_data()

    def _read_and_combine_data(self):
        dataframes = []
        cutoff_date = pd.Timestamp(datetime.now().date()) - pd.DateOffset(years=5)
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
                        df['Ticker'] = file.replace('.csv', '')
                        if df['Date'].max() >= cutoff_date:
                            # Filtering based on the recent five years
                            df = df[df['Date'] >= cutoff_date]
                            # Selective sampling based on 'Close' price
                            percentile_75 = df['Close'].quantile(0.75)
                            df = df[df['Close'] >= percentile_75]
                            dataframes.append(df)
                    except pd.errors.ParserError as e:
                        print(f"Error reading {file_path}: {e}")
        if dataframes:
            combined_data = pd.concat(dataframes, ignore_index=True)
            combined_data.to_parquet(self.processed_file_path)
            return combined_data
        else:
            return pd.DataFrame()


