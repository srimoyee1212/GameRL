# GameRL


# Reinforcement Learning for games (cartpole) using OpenAI Gym (Q-Learning Agent for CartPole-v1 Environment)

This repository contains an implementation of a Q-learning agent to solve the CartPole-v1 environment from OpenAI Gym. The agent is trained using reinforcement learning techniques, specifically Q-learning with an epsilon-greedy policy. The state space is discretized to allow the agent to effectively learn and optimize its actions.

## Table of Contents
- [Introduction](#introduction)
- [Theory](#theory)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)

## Introduction
This project demonstrates the application of Q-learning, a model-free reinforcement learning algorithm, on the CartPole-v1 environment. The goal is to balance a pole on a cart by applying forces to move the cart left or right. The agent learns to perform this task by interacting with the environment and receiving rewards based on its performance.

## Theory
### Reinforcement Learning
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. The key components of RL are:
- **Agent**: The learner or decision-maker.
- **Environment**: The external system with which the agent interacts.
- **State**: A representation of the current situation of the agent in the environment.
- **Action**: A decision or move made by the agent.
- **Reward**: Feedback from the environment based on the action taken by the agent.

### Q-Learning
Q-learning is a value-based RL algorithm that aims to learn the optimal action-selection policy. It uses a Q-table to store Q-values, which represent the expected future rewards for each state-action pair. The Q-learning update rule is as follows:
\[ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a) - Q(s, a) \right] \]
where:
- \( Q(s, a) \) is the current Q-value.
- \( \alpha \) is the learning rate.
- \( r \) is the reward received after taking action \( a \) from state \( s \).
- \( \gamma \) is the discount factor.
- \( s' \) is the next state.

### Epsilon-Greedy Policy
The epsilon-greedy policy balances exploration and exploitation by choosing a random action with probability \( \epsilon \) and the action with the highest Q-value with probability \( 1 - \epsilon \). This helps the agent explore the environment while gradually exploiting the learned knowledge.

### State Discretization
CartPole-v1 has a continuous state space, which is discretized into bins for simplicity. This allows the Q-learning agent to manage the state space more effectively by treating each discretized state as a distinct state.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cartpole-qlearning-agent.git
   cd cartpole-qlearning-agent
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage
```bash
python test_agent.py
```

## Files
1. agent.py: Contains the QLearningAgent class, implementing the Q-learning algorithm.
2. environment.py: Contains the CartPoleEnvironment class, managing the interaction with the CartPole-v1 environment and state discretization.
3. test_agent.py: Main script to train and test the Q-learning agent.
4. requirements.txt: Lists the required Python packages.

