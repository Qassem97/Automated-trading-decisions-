import matplotlib.pyplot as plt
from data_loader import DataLoader
from data_processor import DataProcessor
from model_trainer import ModelTrainer

if __name__ == "__main__":
    directory_path = 'stock_data'
    processed_data_file = 'processed_data.parquet'
    features = ['MA20', 'MA50', 'Volatility', 'Volume_norm']
    target = 'Returns'

    loader = DataLoader(directory_path, processed_data_file)
    combined_data = loader.load_data()

    if not combined_data.empty:
        processor = DataProcessor(combined_data)
        X_train, X_test, y_train, y_test = processor.process_data(features, target)

        trainer = ModelTrainer(X_train, y_train)
        models = trainer.train_models()
        mse_results = trainer.predict_and_evaluate(X_test, y_test)

        # Plotting example for Apple data (assuming 'AAPL' ticker exists)
        apple_data = combined_data[combined_data['Ticker'] == 'AAPL']
        plt.figure(figsize=(10, 6))
        plt.plot(apple_data['Date'], apple_data['MA20'], label='20-day MA')
        plt.title('20-Day Moving Average for Apple')
        plt.xlabel('Date')
        plt.ylabel('Moving Average Price')
        plt.legend()
        plt.show()
    else:
        print("No data available for processing.")
