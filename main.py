import os
import time

import gym
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN, TD3

from real_time_data_processor import RealTimeDataProcessor
from data_loader import DataLoader
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from RDPGModel import RDPGModel
from stock_trading_env import StockTradingEnvHybrid

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def get_unique_tickers(data):
    if 'Ticker' in data.columns:
        return data['Ticker'].unique().tolist()
    else:
        raise ValueError("The DataFrame does not contain a 'Ticker' column")


def setup_environment():
    global directory_path, processed_data_file, sequence_length
    directory_path = 'stock_data'
    processed_data_file = 'processed_data.parquet'
    sequence_length = 10


def load_and_prepare_data():
    loader = DataLoader(directory_path, processed_data_file)
    combined_data = loader.load_data()
    if combined_data.empty:
        print("No data available for processing.")
        return None, None
    processor = DataProcessor(combined_data)
    processor.cnn_feature_extraction(sequence_length)
    processed_data = processor.prepare_data_for_env()
    return train_test_split(processed_data, test_size=0.4, shuffle=False)


def plot_cumulative_returns(data, actual_ticker):
    if data.empty:
        print(f"No data to plot for {actual_ticker}")
    else:
        data = data.copy()  # Make a copy to avoid warnings
        data['Cumulative Returns'] = (1 + data['Returns']).cumprod()
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Cumulative Returns'], label='Cumulative Returns')
        plt.title(f'Cumulative Returns for {actual_ticker}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.show()


# For TD3 and DDQN models
def predict_stable_baselines3(model, observation):
    # Reshape the observation to meet model's expected input shape.
    # If model expects (n_env, n_features) where n_features is 1:
    print(f"Original observation shape: {observation.shape}")
    observation = observation.reshape(-1, 1)  # This reshapes to (n_env, 1) where n_env is len(observation)
    print(f"Reshaped observation shape: {observation.shape}")
    action, _ = model.predict(observation, deterministic=True)
    return action


# For PyTorch Models (RDPG) (Error here -->  Error processing data or making prediction: input.size(-1) must be equal to input_size. Expected 1, got 11 )
def predict_rdpg(model, observation, hidden_state=None):
    model.eval()
    with torch.no_grad():
        print(f"Original observation shape: {observation.shape}")
        # Ensure the observation is correctly shaped as [1, 1, number_of_features]
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 11]
        print(f"Reshaped observation shape: {observation_tensor.shape}")
        if hidden_state is None:
            hidden_state = (torch.zeros(1, 1, 50), torch.zeros(1, 1, 50))

        q_values, new_hidden_state = model(observation_tensor, hidden_state)
        action = q_values.squeeze().numpy()
    return action, new_hidden_state


def predict_model(model, observation, model_name, hidden_state=None):
    if model_name in ['DDQN', 'TD3']:
        return predict_stable_baselines3(model, observation)
    elif model_name == 'RDPG':
        # Properly handle hidden state in RDPG predictions
        action, new_hidden_state = predict_rdpg(model, observation, hidden_state)
        return action  # Return action if hidden state management outside is not needed
    else:
        raise ValueError("Unsupported model type")


def load_stable_baselines3_model(model_class, path):
    try:
        model = model_class.load(path)
        print(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"No saved model found at {path}. A new model will be trained.")
        return None


# Saving and loading for stable_baselines3 models (DDQN, TD3)
def save_stable_baselines3_model(model, path):
    model.save(path)
    print(f"Model saved to {path}")


# Loading RDPG model
def load_pytorch_model(model_class, path, input_dim, hidden_dim, output_dim):
    model = model_class(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model


# Saving RDPG model
def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def train_and_save_model(trainer, model_name, model_class):
    model_path = f"{model_name.lower()}_model.zip"
    if os.path.exists(model_path):
        return load_stable_baselines3_model(model_class, model_path)
    print(f"Starting {model_name} model training...")
    model = getattr(trainer, f'train_{model_name.lower()}')()
    if model:
        save_stable_baselines3_model(model, model_path)
        print(f"{model_name} model training completed.")
    else:
        print(f"{model_name} model training failed.")
    return model


def train_and_save_rdpg_model(trainer):
    model_path = "rdpg_model.zip"
    # Initialize environment to get dimensions
    env = StockTradingEnvHybrid(trainer.data, trainer.tickers, trainer.sequence_length, True)
    input_dim = env.observation_space.shape[0]
    hidden_dim = 50
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        output_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError("Unsupported action space")

    if os.path.exists(model_path):
        return load_pytorch_model(RDPGModel, model_path, input_dim, hidden_dim, output_dim)
    print("Starting RDPG model training...")
    model = trainer.train_rdpg()
    if model:
        save_pytorch_model(model, model_path)
        print("RDPG model training completed.")
    else:
        print("RDPG model training failed.")
    return model


def train_models(train_data, tickers):
    trainer = ModelTrainer(train_data, tickers, sequence_length)
    models = {
        # 'DDQN': train_and_save_model(trainer, 'DDQN', DQN),
        # 'TD3': train_and_save_model(trainer, 'TD3', TD3),
        'RDPG': train_and_save_rdpg_model(trainer)
    }
    return models


def evaluate_models(models, test_data, tickers):
    for model_name, model in models.items():
        if model:
            evaluator = ModelEvaluator(model, test_data)
            evaluator.evaluate()
            for ticker in tickers:
                evaluator.plot_results(ticker)
                plot_cumulative_returns(test_data[test_data['Ticker'] == ticker], ticker)


def continuous_real_time_prediction(models, ticker):
    real_time_processor = RealTimeDataProcessor()
    model_names = list(models.keys())
    hidden_states = {model_name: None for model_name in model_names if
                     'RDPG' in model_name}  # Initialize hidden states for RDPG models

    while True:
        try:
            current_data = real_time_processor.fetch_and_process(ticker)
            if 'Date' in current_data.columns:
                current_data = current_data.drop(columns=['Date'])
            observation = current_data.values.flatten()

            for model_name in model_names:
                if models[model_name] is not None:
                    if model_name == 'RDPG':
                        # Handle stateful prediction for RDPG
                        action, hidden_states[model_name] = predict_model(models[model_name], observation, model_name,
                                                                          hidden_states[model_name])
                    else:
                        # Stateless prediction for other models
                        action = predict_model(models[model_name], observation, model_name)

                    print(
                        f"Recommended action for the ticker {ticker} by {model_name} based on the latest data: {action}")

        except Exception as e:
            print(f"Error processing data or making prediction: {str(e)}")
        time.sleep(60)  # Sleep for 60 seconds


def main():
    setup_environment()
    train_data, test_data = load_and_prepare_data()
    if train_data is not None:
        train_tickers = get_unique_tickers(train_data)
        test_tickers = get_unique_tickers(test_data)
        # To test otherwise take the defined above (test_tickers).
        tickers = ['ARI', 'ZNH']
        models = train_models(train_data, train_tickers)
        evaluate_models(models, test_data, tickers)
        continuous_real_time_prediction(models, "ARI")
    else:
        print("Failed to load or process data.")


if __name__ == "__main__":
    main()
