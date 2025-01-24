import numpy as np
import pandas as pd
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random

class PriceOptimizationEnv(gym.Env):
    def __init__(self, historical_data: pd.DataFrame):
        super().__init__()
        self.data = historical_data.dropna(subset=['Product Price', 'Total Sales', 'Predicted Sales'])
        
        self.min_price = self.data['Product Price'].min()
        self.max_price = self.data['Product Price'].max()
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([self.max_price, 1, 1]), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([self.min_price]), 
            high=np.array([self.max_price]), 
            dtype=np.float32
        )
        
        self.price_scaler = MinMaxScaler()
        self.price_scaler.fit(self.data[['Product Price']])
        
        self.sales_scaler = MinMaxScaler()
        self.sales_scaler.fit(np.column_stack([
            self.data['Total Sales'], 
            self.data['Predicted Sales']
        ]))
        
        self.current_step = 0
        self.max_steps = len(self.data)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        initial_price = self.data.iloc[0]['Product Price']
        initial_sales = self.data.iloc[0]['Total Sales']
        initial_predicted_sales = self.data.iloc[0]['Predicted Sales']
        
        state = np.array([
            self.price_scaler.transform([[initial_price]])[0][0],
            self.sales_scaler.transform([[initial_sales, initial_predicted_sales]])[0][0],
            self.sales_scaler.transform([[initial_sales, initial_predicted_sales]])[0][1]
        ])
        
        return state, {}
    
    def step(self, action):
        current_price = self.price_scaler.inverse_transform(action.reshape(-1, 1))[0][0]
        
        closest_idx = np.abs(self.data['Product Price'] - current_price).argmin()
        historical_record = self.data.iloc[closest_idx]
        
        total_sales = historical_record['Total Sales']
        predicted_sales = historical_record['Predicted Sales']
        organic_conversion = historical_record.get('Organic Conversion Percentage', 0)
        ad_conversion = historical_record.get('Ad Conversion Percentage', 0)
        
        reward = (
            (total_sales / predicted_sales) * 0.5 +  
            (organic_conversion / 100) * 0.25 +      
            (ad_conversion / 100) * 0.25 +           
            (current_price / self.max_price) * 0.5   
        )
        
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        
        next_state = np.array([
            self.price_scaler.transform([[current_price]])[0][0],
            self.sales_scaler.transform([[total_sales, predicted_sales]])[0][0],
            self.sales_scaler.transform([[total_sales, predicted_sales]])[0][1]
        ])
        
        return next_state, reward, done, False, {}

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return np.random.uniform(low=0, high=1, size=(1,))
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.numpy()

def train_price_optimization_model(data_path):
    # Load data
    data = pd.read_csv(data_path)
    
    # Create environment
    env = PriceOptimizationEnv(data)
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.shape[0]
    )
    
    # Training loop
    num_episodes = 1000
    best_reward = float('-inf')
    best_price = None
    
    print("Starting Price Optimization Training...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            state = next_state
        
        # Track best price strategy
        if total_reward > best_reward:
            best_reward = total_reward
            best_price = action[0]
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {total_reward:.2f}")
    
    # Final results
    print("\nTraining Complete!")
    print(f"Best Price Strategy: ${best_price:.2f}")
    print(f"Best Total Reward: {best_reward:.2f}")
    
    return agent, best_price, best_reward

# Uncomment to run
agent, best_price, best_reward = train_price_optimization_model('soapnutshistory.csv')