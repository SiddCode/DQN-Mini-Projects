import gymnasium as gym

def run():

    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode='human')

    state = env.reset()[0]
    terminated = False # Falls into ice holes or goal
    truncated = False # if the agent is more than 200 steps

    while(not terminated and not truncated):

        # Actions are up,down,left,right
        action = env.action_space.sample()

        new_state,reward,terminated,truncated,_ = env.step(action)

        state = new_state

    env.close()

if __name__ == '__main__':
    run()