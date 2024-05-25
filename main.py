import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator


def plot_cumulative_returns(data, actual_ticker):
    data['Cumulative Returns'] = (1 + data['Returns']).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Cumulative Returns'], label='Cumulative Returns')
    plt.title(f'Cumulative Returns for {actual_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    directory_path = 'stock_data'
    processed_data_file = 'processed_data.parquet'
    tickers = ['AAPL', 'MSFT']  # List of tickers to use
    sequence_length = 30  # Define the sequence lengths to test

    # Load and preprocess data
    loader = DataLoader(directory_path, processed_data_file)
    combined_data = loader.load_data()  # This should return a DataFrame

    if not combined_data.empty:
        print("Loaded Data Summary:")
        print(combined_data.head())

        processor = DataProcessor(combined_data)

        # CNN feature extraction
        processor.cnn_feature_extraction(sequence_length)
        # preprocess the data
        processed_data = processor.prepare_data_for_env()

        print("Data after CNN feature extraction and preprocessing:")
        print(processed_data.head())

        # Split the processed_data
        train_data, test_data = train_test_split(processed_data, test_size=0.4, shuffle=False, stratify=None)

        print("Processed Training Data Summary:")
        print(train_data.head())

        print("Processed Testing Data Summary:")
        print(test_data.head())

        # Train the model
        trainer = ModelTrainer(train_data, tickers, sequence_length)

        # Train DDQN model
        print("Starting DDQN model training...")
        ddqn_model = trainer.train_ddqn_model()
        if ddqn_model:
            print("DDQN model training completed.")
        else:
            print("DDQN model training failed.")

        """
        # Train TD3 model
        print("Starting TD3 model training...")
        td3_model = trainer.train_td3_model()
        if td3_model:
            print("TD3 model training completed.")
        else:
            print("TD3 model training failed.")

        # Train RDPG model
        print("Starting RDPG model training...")
        rdpg_model = trainer.train_rdpg_model()
        if rdpg_model:
            print("RDPG model training completed.")
        else:
            print("RDPG model training failed.")
        """

        # Evaluate models
        if ddqn_model:
            ddqn_evaluator = ModelEvaluator(ddqn_model, test_data)
            ddqn_evaluator.evaluate()

            for ticker in tickers:
                ddqn_evaluator.plot_results(ticker)
                plot_cumulative_returns(test_data[test_data['Ticker'] == ticker], ticker)
        else:
            print("DDQN model training failed or returned None.")

        """
        if td3_model:
            td3_evaluator = ModelEvaluator(td3_model, test_data)
            td3_evaluator.evaluate()

            for ticker in tickers:
                td3_evaluator.plot_results(ticker)
                plot_cumulative_returns(test_data[test_data['Ticker'] == ticker], ticker)
        else:
            print("TD3 model training failed or returned None.")

        if rdpg_model:
            rdpg_evaluator = ModelEvaluator(rdpg_model, test_data)
            rdpg_evaluator.evaluate()

            for ticker in tickers:
                rdpg_evaluator.plot_results(ticker)
                plot_cumulative_returns(test_data[test_data['Ticker'] == ticker], ticker)
        else:
            print("RDPG model training failed or returned None.")
            """
    else:
        print("No data available for processing.")
