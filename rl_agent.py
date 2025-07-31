import os
import gymnasium as gym
from trading_env import TradingEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# --- CONFIGURATION ---
DATA_CSV = 'merged_features_daily.csv'  # Change to 'merged_features_hourly.csv' for hourly
MODEL_DIR = 'rl_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- ENVIRONMENT ---
env = TradingEnv(DATA_CSV)

# --- DQN AGENT ---
print('Training DQN agent...')
dqn = DQN('MlpPolicy', env, verbose=1)
dqn.learn(total_timesteps=10000)
dqn.save(os.path.join(MODEL_DIR, 'dqn_trading'))

# --- PPO AGENT ---
print('Training PPO agent...')
ppo = PPO('MlpPolicy', env, verbose=1)
ppo.learn(total_timesteps=10000)
ppo.save(os.path.join(MODEL_DIR, 'ppo_trading'))

# --- EVALUATION ---
print('Evaluating DQN agent...')
dqn_env = TradingEnv(DATA_CSV)
dqn_mean_reward, dqn_std_reward = evaluate_policy(dqn, dqn_env, n_eval_episodes=5, return_episode_rewards=False)
print(f'DQN Mean Reward: {dqn_mean_reward:.2f} +/- {dqn_std_reward:.2f}')

print('Evaluating PPO agent...')
ppo_env = TradingEnv(DATA_CSV)
ppo_mean_reward, ppo_std_reward = evaluate_policy(ppo, ppo_env, n_eval_episodes=5, return_episode_rewards=False)
print(f'PPO Mean Reward: {ppo_mean_reward:.2f} +/- {ppo_std_reward:.2f}')
