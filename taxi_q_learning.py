# Taxi Q Learning

import gymnasium as gym
import numpy as np


def run(episodes):
    # Initializes the Environment
    env = gym.make('Taxi-v3', render_mode='human')

    # Create a Q-Learning Table
    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 500 x 6 table

    # Define Parameters
    learning_rate = 0.9 
    discount_factor = 0.92
    epsilon = 0.05
    random_number = np.random.default_rng()

    for i in range(episodes):
        # Starting values
        state = env.reset()[0] 
        terminated = False 
        truncated = False  
         
        while(not terminated and not truncated):
            # Use Epsilon-Greedy Exploration
            if random_number.random() < epsilon:
                # Random Action
                action = env.action_space.sample() 
                print('------------------------')
                print('random action')
            else:
                action = np.argmax(q[state,:])
                print('greedy action')

            next_state,reward,terminated,truncated,info = env.step(action)

            print(next_state,reward,terminated,truncated)
            print(q[state,action])
            print('------------------------')


            q[state,action] = q[state,action] + learning_rate * (reward + discount_factor * np.max(q[next_state,:]) - q[state,action])

            state = next_state

    env.close()

    


if __name__ == '__main__':
    run(3)