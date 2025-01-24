# Price Optimization with Reinforcement Learning
This project implements a **Price Optimization** solution using a Reinforcement Learning (RL) approach. The agent learns optimal pricing strategies by interacting with an environment modeled as a **Markov Decision Process (MDP)**. The agent uses a **Q-Network** to approximate the action-value function and is trained using Deep Q-Learning.
---
## Features
- **Custom Environment**: Simulates a pricing scenario where the agent decides on optimal prices based on environment feedback.
- **Deep Q-Network**: A feed-forward neural network for estimating Q-values for all actions in a given state.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation during training.
- **Flexible Configuration**: Parameters such as learning rate, discount factor, and epsilon decay are adjustable.
- **Evaluation Mode**: Test the performance of the trained agent on unseen data.
---
## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- `pip` for package management
### Clone the Repository
```bash
git clone https://github.com/your-username/price-optimization-rl.git
cd price-optimization-rl
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
---
## Project Structure
```
price-optimization-rl/
│
├── data/
│   └── soapnutshistory.csv         # Sample historical sales data
│
├── src/
│   ├── __init__.py
│   ├── environment.py               # Custom Gym environment
│   ├── agent.py                     # Deep Q-Network agent implementation
│   └── train.py                     # Training script
│
├── notebooks/
│   └── analysis.ipynb               # Jupyter notebook for data analysis
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
└── main.py                          # Main execution script
```
---
## Usage
### Training the Model
```python
from src.train import train_price_optimization_model

# Train the model using historical sales data
agent, best_price, best_reward = train_price_optimization_model('data/soapnutshistory.csv')

# Print the optimal pricing strategy
print(f"Optimal Price: ${best_price:.2f}")
print(f"Best Reward: {best_reward:.2f}")
```
---
## Methodology
### Reinforcement Learning Approach
1. **Environment Creation**: A custom OpenAI Gym environment is designed to simulate pricing scenarios.
2. **State Representation**: Includes normalized price, total sales, and predicted sales.
3. **Action Space**: Continuous price selection within historical price ranges.
4. **Reward Calculation**: Multi-factor reward function considering:
   - Sales performance
   - Conversion rates
   - Pricing strategy
5. **Agent Learning**: Deep Q-Network approximates the optimal pricing policy.
---
## Technical Details
- **Framework**: Gymnasium, PyTorch
- **Learning Algorithm**: Deep Q-Learning
- **Neural Network**: 
  - Input Layer: Normalized state features
  - Hidden Layers: Two layers with ReLU activation
  - Output Layer: Q-value estimation
---
## Performance Metrics
- **Total Reward**: Aggregate performance score
- **Optimal Price**: Best identified pricing point
- **Exploration vs Exploitation**: Balanced through epsilon-greedy strategy
---
## Limitations
- Requires high-quality historical sales data
- Assumes linear relationships in sales dynamics
- Performance depends on data representativeness
---
## Future Improvements
- Implement experience replay
- Add more complex reward functions
- Support multi-product optimization
- Enhanced feature engineering
---
## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
---
Project Link: [https://github.com/dhruvpal102005/Price-Optimization.git](https://github.com/dhruvpal102005/Price-Optimization.git)
