import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom trading environment for RL and simulation.
    State: [sentiment features, price, position, cash]
    Action: 0=hold, 1=buy, 2=sell
    Reward: change in portfolio value (net of transaction costs)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_csv, initial_cash=10000, transaction_cost=0.001, slippage=0.001):
        super(TradingEnv, self).__init__()
        self.data = pd.read_csv(data_csv)
        self.features = ['avg_sentiment', 'sentiment_volatility', 'post_volume', 'sentiment_change', 'Close']
        self.n_steps = len(self.data)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.action_space = gym.spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        # obs: features + position + cash
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features) + 2,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0  # number of shares
        self.last_price = self.data.iloc[self.current_step]['Close']
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        obs = [row[f] for f in self.features] + [self.position, self.cash]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}
        row = self.data.iloc[self.current_step]
        price = row['Close']
        reward = 0.0
        info = {}
        # Transaction cost and slippage
        cost = 0.0
        if action == 1:  # buy
            if self.cash >= price:
                shares = int(self.cash // price)
                exec_price = price * (1 + self.slippage)
                cost = exec_price * shares * self.transaction_cost
                self.cash -= exec_price * shares + cost
                self.position += shares
        elif action == 2:  # sell
            if self.position > 0:
                exec_price = price * (1 - self.slippage)
                cost = exec_price * self.position * self.transaction_cost
                self.cash += exec_price * self.position - cost
                self.position = 0
        # Reward: change in portfolio value
        portfolio_value = self.cash + self.position * price
        reward = portfolio_value - (self.cash + self.position * self.last_price)
        self.last_price = price
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            self.done = True
        return self._get_obs(), reward, self.done, False, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Price: {self.last_price:.2f}, Position: {self.position}, Cash: {self.cash:.2f}")

    def close(self):
        pass

# Example usage:
# env = TradingEnv('merged_features_daily.csv')
# obs, _ = env.reset()
# for _ in range(100):
#     action = env.action_space.sample()
#     obs, reward, done, _, _ = env.step(action)
#     env.render()
#     if done:
#         break 