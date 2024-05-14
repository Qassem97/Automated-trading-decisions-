import pandas as pd
import os

class DataLoader:
    def __init__(self, directory_path, processed_file_path):
        self.directory_path = directory_path
        self.processed_file_path = processed_file_path

    def load_data(self):
        if os.path.exists(self.processed_file_path):
            print(f"Loading data from {self.processed_file_path}")
            return pd.read_parquet(self.processed_file_path)
        else:
            return self._read_and_combine_data()

    def _read_and_combine_data(self):
        dataframes = []
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
                        df['Ticker'] = file.replace('.csv', '')
                        dataframes.append(df)
                    except pd.errors.ParserError as e:
                        print(f"Error reading {file_path}: {e}")
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()
