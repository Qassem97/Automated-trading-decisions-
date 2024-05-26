import gym
from gym import spaces
import numpy as np


class StockTradingEnvHybrid(gym.Env):
    def __init__(self, df, tickers, sequence_length, continuous_action):
        super(StockTradingEnvHybrid, self).__init__()
        self.df = df.dropna(subset=['CNN_Features'])
        self.tickers = tickers
        self.sequence_length = sequence_length
        self.continuous_action = continuous_action
        self.current_ticker_index = 0
        self.current_ticker = self.tickers[self.current_ticker_index]
        self.current_step = 0
        self.done = False
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        feature_length = len(self.df['CNN_Features'].iloc[0])  # Ensure this is 11
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_length,), dtype=np.float32)
        if self.continuous_action:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        if self.current_ticker_index < len(self.tickers) - 1:
            self.current_ticker_index += 1
        else:
            self.current_ticker_index = 0
        self.current_ticker = self.tickers[self.current_ticker_index]
        print(f"Resetting environment for ticker {self.current_ticker}")
        return self._next_observation()

    def _next_observation(self):
        filtered_df = self.df[self.df['Ticker'] == self.current_ticker]
        if self.current_step >= len(filtered_df):
            print("No more data available for this ticker, resetting...")
            self.done = True
            return np.zeros(self.observation_space.shape)  # Return a zero array when out of data

        obs = filtered_df.iloc[self.current_step]['CNN_Features']
        if isinstance(obs, (list, np.ndarray)):
            return np.array(obs)
        else:
            raise ValueError("CNN_Features column must contain lists or arrays")

    def step(self, action):
        if self.done:
            return np.zeros(self.observation_space.shape), 0, self.done, {}

        if self.continuous_action:
            self._take_action_continuous(action[0])
        else:
            self._take_action_discrete(action)

        self.current_step += 1
        if self.current_step >= len(self.df[self.df['Ticker'] == self.current_ticker]):
            print("Reached the end of data for the current ticker.")
            self.done = True
            return np.zeros(self.observation_space.shape), 0, self.done, {}

        reward = self._calculate_reward()
        obs = self._next_observation()

        return obs, reward, self.done, {}

    def _take_action_discrete(self, action):
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

    def _take_action_continuous(self, action):
        current_price = self.df[self.df['Ticker'] == self.current_ticker].iloc[self.current_step]['Close']
        if action > 0:
            spend = self.balance * min(1, action)
            shares_to_buy = spend / current_price
            self.shares_held += shares_to_buy
            self.balance -= spend
        elif action < 0:
            shares_to_sell = self.shares_held * min(1, -action)
            self.shares_held -= shares_to_sell
            self.balance += shares_to_sell * current_price

    def _calculate_reward(self):
        current_price = self.df[self.df['Ticker'] == self.current_ticker].iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.shares_held * current_price
        reward = portfolio_value - self.initial_balance
        return reward
