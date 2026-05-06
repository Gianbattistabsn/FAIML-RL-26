"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
from agent import Agent, Policy

NUM_EPISODES = 100
SEED = 42
RENDER = True


def main():
    env = gym.make('Hopper-v4', render_mode='human' if RENDER else None)    
    policy = Policy(state_space=env.observation_space.shape[0], action_space = env.action_space.shape[0])
    policy.load_state_dict(torch.load("part1/checkpoints/reinforce_baseline_20_2026-05-06_20-59-46_ep20000.pt", weights_only=False))
    policy.eval()
    # it automatically sets the device if available
    agent = Agent(policy)

    for i in range(NUM_EPISODES):
        done = False
        state, info = env.reset(seed=SEED + i)  # vary seed per episode for different starts
        # keep running until the episode is done
        while not done:
            
            action, action_log_prob = agent.get_action(state, evaluation=True)
            
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())

            # The episode is over if EITHER termination condition is met.
            done = terminated or truncated

            # No need to call env.render() manually: with render_mode='human'
            # gymnasium renders automatically on every env.step()
            state = next_state

    env.close()

if __name__ == '__main__':
    main()