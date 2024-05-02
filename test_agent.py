from environment import CartPoleEnvironment
from agent import QLearningAgent
import numpy as np

def test_agent():
    env = CartPoleEnvironment()
    agent = QLearningAgent(state_space_size=env.env.observation_space.shape[0],
                           action_space_size=env.env.action_space.n)

    num_episodes = 1000
    max_steps = 200
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done= env.step(action)  # Ensure only three values are unpacked
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            if done:
                break
        agent.decay_epsilon()
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

test_agent()
