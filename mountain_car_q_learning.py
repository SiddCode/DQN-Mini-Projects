# Mountain Car Q Learning

import gymnasium as gym
import numpy as np


def run(episodes):
    # Initializes the Environment
    env = gym.make('MountainCar-v0', render_mode='human')

    # Divide Map into states
    position = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    velocity = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    q = np.zeros((len(position), len(velocity), env.action_space.n)) # 20x20x3 table

    # Define Parameters
    learning_rate = 0.9 
    discount_factor = 0.9
    epsilon = 0.05
    random_number = np.random.default_rng()

    for i in range(episodes):

        state = env.reset()[0] 
        position_state = np.digitize(state[0], position)
        velocity_state = np.digitize(state[1], velocity)

        terminated = False 
            
        while(not terminated):
            # # Use Epsilon-Greedy Exploration
            if random_number.random() < epsilon:
                # Random Action
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[position_state, velocity_state,:])


            next_state,reward,terminated,truncated,info = env.step(action)
            next_position_state = np.digitize(next_state[0], position)
            next_velocity_state = np.digitize(next_state[1], velocity)

            q[position_state,velocity_state,action] = q[position_state,velocity_state,action] + learning_rate * (reward + discount_factor * np.max(q[next_position_state,next_velocity_state,:]) - q[position_state,velocity_state,action])


            state = next_state
            position_state = next_position_state
            velocity_state = next_velocity_state

    env.close()

    


if __name__ == '__main__':
    run(10)