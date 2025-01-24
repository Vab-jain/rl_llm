import numpy as np 
import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv  
from stable_baselines3.common.env_checker import check_env 
from pettingzoo.classic import tictactoe_v3 
from gymnasium import spaces 
from time import sleep


class TicTacToeCustomEnv(gym.Env): 
    """ 
    Custom Tic-Tac-Toe Environment for Reinforcement Learning. 
    This environment is compatible with Stable Baselines3. 
    """ 
    def __init__(self, render_mode=None): 
        super(TicTacToeCustomEnv, self).__init__() 
        # Create the PettingZoo Tic-Tac-Toe environment 
        self.env = tictactoe_v3.env(render_mode=render_mode) 
        
        # Define action and observation space 
        # There are 9 possible actions (places to mark on the board) 
        self.action_space = spaces.Discrete(9)    

        # Observation space: 9 board positions + action mask of size 9 (legal moves) 
        # We'll flatten both and concatenate them into a single vector for the observation. 
        self.observation_space = spaces.Dict({ 
            "observation": spaces.Box(low=0, high=1, shape=(3,3,2), dtype=np.int8), 
            "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8) 
        })
  
    def reset(self, seed=None, options=None): 
        # Set the seed if provided 
        super().reset(seed=seed) 
        self.env.reset(seed=seed) 
         
        # Get the initial observation 
        observation, _, _, _, _ = self.env.last() 
         
        # Return observation and an empty info dictionary (required by Gym API) 
        return observation, {} 
  
    def step(self, action): 
        # Get the current agent 
        agent = self.env.agent_selection 
        observation, reward, termination, truncation, info = self.env.last() 
         
        if termination or truncation: 
            # End the episode if the game is over 
            return observation, reward, termination, truncation, info 
  
        # Apply the action mask 
        mask = observation["action_mask"] 
         
        if mask[action] == 0:  # Illegal move 
            reward = -1 
            termination = True 
        else: 
            self.env.step(action) 
            observation, reward, termination, truncation, info = self.env.last() 
  
        return observation, reward, termination, truncation, info 
  
    def render(self): 
        self.env.render() 
  
    def close(self): 
        self.env.close() 
        

mode = 'test'

if mode == 'train':
    # Initialize the custom environment 
    env = TicTacToeCustomEnv() 
    
    # Check if the environment follows the OpenAI Gym API 
    check_env(env, warn=True) 
    
    # Create and wrap the environment 
    vec_env = DummyVecEnv([lambda: env]) 
    
    # Define the model 
    model = PPO("MultiInputPolicy", vec_env, verbose=1) 
    
    # Train the model 
    model.learn(total_timesteps=10000) 
    
    # Save the model 
    model.save("ppo_tictactoe") 
    
elif mode == 'test':
    # To load the model    later 
    model = PPO.load("ppo_tictactoe") 
  
# # Testing the trained model 
# test_env = TicTacToeCustomEnv(render_mode="human")
# test_env = DummyVecEnv([lambda: test_env]) 

# obs = test_env.reset() 
# for i in range(1000): 
#     action, _states = model.predict(obs, deterministic=True) 
#     obs, rewards, dones, info = test_env.step(action) 
#     test_env.render() 
#     if dones: 
#         obs = test_env.reset() 
    
    
# test_env.close() 
 
 # Playing against the random bot in testing 

test_env = TicTacToeCustomEnv(render_mode=None) 
obs, _ = test_env.reset() 


while True: 
    action, _states = model.predict(obs, deterministic=True) 
    obs, reward, done, info = test_env.step(action) 

    if done: 
        print("Game over! Reward:", reward) 
        break 
    # Random bot's turn 
    bot_action = random_bot(obs) 
    obs, reward, done, info = test_env.step(bot_action) 
     

    if done: 
        print("Game over! Bot reward:", reward) 
        break 

test_env.close() 


for agent in env.agent_iter(): 

    observation, reward, termination, truncation, info = env.last() 

  

    if termination or truncation: 

        action = None 

    else: 

        mask = observation["action_mask"] 

        # For the RL agent (using PPO), the policy is applied here 

        action = env.action_space(agent).sample(mask)  # <-- Random bot when it's its turn 

  

    env.step(action) 
 