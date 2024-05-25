import gym
from gym import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self, df, tickers, sequence_length):
        super(StockTradingEnv, self).__init__()
        self.df = df.dropna(subset=['CNN_Features'])  # Drop rows where CNN_Features is NaN
        self.tickers = tickers
        self.sequence_length = sequence_length
        self.current_ticker = self.tickers[0]  # Start with the first ticker
        self.current_step = 0
        self.done = False
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Ensure the CNN_Features column is a list or array
        feature_length = len(self.df['CNN_Features'].iloc[0])
        self.action_space = spaces.Discrete(3)  # Discrete action space (buy, sell, hold)
        self.observation_space = spaces.Box(low=0, high=1, shape=(feature_length,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_ticker = self.tickers[0]  # Reset to the first ticker
        self.done = False
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        print(f"Resetting environment for ticker {self.current_ticker}")
        return self._next_observation()

    def _next_observation(self):
        filtered_df = self.df[self.df['Ticker'] == self.current_ticker]
        if self.current_step >= len(filtered_df):
            self.done = True
            return np.zeros(self.observation_space.shape)

        obs = filtered_df.iloc[self.current_step]['CNN_Features']
        if isinstance(obs, (list, np.ndarray)):
            print(f"Observation at step {self.current_step} for ticker {self.current_ticker}: {obs}")
            return np.array(obs)
        else:
            raise ValueError("CNN_Features column must contain lists or arrays")

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.df[self.df['Ticker'] == self.current_ticker]):
            self.done = True

        reward = self._calculate_reward()
        obs = self._next_observation() if not self.done else np.zeros(self.observation_space.shape)

        return obs, reward, self.done, {}

    def _take_action(self, action):
        current_price = self.df[self.df['Ticker'] == self.current_ticker].iloc[self.current_step]['Close']

        if action == 0:  # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 1:  # Sell
            self.shares_held -= 1
            self.balance += current_price
            self.total_shares_sold += 1
            self.total_sales_value += current_price
        # action == 2 is hold, no change

    def _calculate_reward(self):
        current_price = self.df[self.df['Ticker'] == self.current_ticker].iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.shares_held * current_price
        reward = portfolio_value - self.initial_balance
        return reward
