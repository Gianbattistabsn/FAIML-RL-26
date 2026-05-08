"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import time
import gymnasium as gym
import panda_gym # type: ignore[import-not-found]

def main():
    render = True

    env = gym.make(
        "PandaPush-v3",
        render_mode="human" if render else "rgb_array",
        type="target",
        reward_type="dense",
    )
    
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = 1000

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        cumsum = 0
        while not done:  # Until the episode is over
            time.sleep(0.02)
            action = env.action_space.sample()  # Sample random action

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated

            cumsum += reward

            if render:
                env.render()
        
        print(f"\nreturn for episode {ep+1} = {cumsum}\n")


if __name__ == '__main__':
    main()