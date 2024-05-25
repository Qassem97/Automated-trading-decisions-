import gym
from stable_baselines3 import DQN, TD3  # DDQN can be handled via DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from RDPGModel import RDPGModel, RDPGTrainer
from stock_trading_env import StockTradingEnv


class ModelTrainer:
    def __init__(self, data, tickers, sequence_length, learning_rate=0.0005, verbose=1):
        self.data = data
        self.tickers = tickers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.verbose = verbose

    def train_ddqn_model(self, total_timesteps=10000):
        try:
            env = DummyVecEnv([lambda: StockTradingEnv(self.data, self.tickers, self.sequence_length)])
            model = DQN('MlpPolicy', env, learning_rate=self.learning_rate, verbose=self.verbose)
            model.learn(total_timesteps=total_timesteps)
            return model
        except Exception as e:
            print(f"Failed to train DDQN model: {str(e)}")
            return None

    def train_td3_model(self, total_timesteps=10000):
        try:
            env = DummyVecEnv([lambda: StockTradingEnv(self.data, self.tickers, self.sequence_length)])
            model = TD3('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=total_timesteps)
            return model
        except Exception as e:
            print(f"Failed to train TD3 model: {str(e)}")
            return None

    def train_rdpg_model(self):
        try:
            env = StockTradingEnv(self.data, self.tickers, self.sequence_length)
            input_dim = env.observation_space.shape[0]

            if isinstance(env.action_space, gym.spaces.Discrete):
                output_dim = env.action_space.n  # Number of discrete actions
            elif isinstance(env.action_space, gym.spaces.Box):
                output_dim = env.action_space.shape[0]  # Dimensionality of the action vector
            else:
                raise NotImplementedError("Unsupported action space")

            model = RDPGModel(input_dim, 50, output_dim)
            target_model = RDPGModel(input_dim, 50, output_dim)
            trainer = RDPGTrainer(env, model, target_model)
            trainer.train()
            return model if model.is_trained() else None
        except Exception as e:
            print(f"Failed to train RDPG model: {str(e)}")
            return None
