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
    policy.load_state_dict(torch.load("part1/checkpoints/reinforce_2026-05-06_19-39-05_16648.558904650956_500_33.29711780930191", weights_only=False))
    policy.eval()
    # it automatically sets the device if available
    agent = Agent(policy)

    for i in range(NUM_EPISODES):
        done = False
        state, info = env.reset()
        # keep running until the episode is done
        while not done:
            
            action, action_log_prob = agent.get_action(state, evaluation=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())

            # The episode is over if EITHER termination condition is met.
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            if RENDER:
                env.render()   # refresh the visual window
            state = next_state

if __name__ == '__main__':
    main()