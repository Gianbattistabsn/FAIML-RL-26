"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import datetime

import numpy as np
import gymnasium as gym
import torch
from agent import Agent, Policy

algorithm = 'reinforce'
baseline = 20
run_name = 'baseline_20'        # change this to label different runs (e.g. 'no_baseline')
NUM_EPISODES = 20000
CHECKPOINT_INTERVAL = 5000     # save a checkpoint every N episodes
SEED = 42
RENDER = False
PATH = f"part1/checkpoints/{algorithm}_{run_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"


def main():
    # set all seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make('Hopper-v4', render_mode='human' if RENDER else None)    
    policy = Policy(state_space=env.observation_space.shape[0], action_space = env.action_space.shape[0])
    print("PATH IS", PATH)
    
    total_reward = 0.0

    # it automatically sets the device if available
    agent = Agent(policy)

    for i in range(NUM_EPISODES):
        done = False
        state, info = env.reset(seed=SEED + i)
        # keep running until the episode is done
        while not done:
            
            action, action_log_prob = agent.get_action(state, evaluation=False)

            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            total_reward += reward
            # The episode is over if EITHER termination condition is met.
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, reward, done)
            agent.update_policy(algorithm=algorithm, baseline=baseline)

            if RENDER:
                env.render()   # refresh the visual window
            state = next_state

        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}/{NUM_EPISODES} finished")

        # save intermediate checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = PATH + f'_ep{i+1}.pt'
            torch.save(policy.state_dict(), ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

    final_path = PATH + f'_{total_reward:.0f}_{NUM_EPISODES}_{total_reward/NUM_EPISODES:.1f}.pt'
    torch.save(policy.state_dict(), final_path)


if __name__ == '__main__':
    main()