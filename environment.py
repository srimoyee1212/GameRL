import gym
import numpy as np

class CartPoleEnvironment:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.bins = [
            # Define bins for each dimension of the state space
            [-2.4, 2.4],
            [-3.0, 3.0],
            [-0.5, 0.5],
            [-2.0, 2.0]
        ]

    def reset(self):
        state = self.env.reset()
        print("Reset state:", state)
        self.env.render()
        return self.discretize(state)

    def step(self, action):
        next_state, reward, done, _, item5 = self.env.step(action)
        print("Next state:", next_state)
        self.env.render()
        return self.discretize(next_state), reward, done

    def discretize(self, state):
        print(type(state))  # Add this line to check the type of state
    # Check if state is a tuple
        if isinstance(state, tuple):
            state_array = state[0]  # Unpack the state tuple
            print("State array:", state_array)
        else:
            state_array = state
            print("State array:", state_array)
    
    # Discretize each dimension of the state space
        discretized_state = []
        for i in range(len(self.bins)):
            bins = self.bins[i]
            print(type(bins))
            print(state_array[i])
            discretized_state.append(
                np.digitize(state_array[i], bins) - 1  # Subtract 1 to adjust index to start from 0
            )
        return tuple(discretized_state)






