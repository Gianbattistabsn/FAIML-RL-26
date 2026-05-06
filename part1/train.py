"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import datetime
import os

import numpy as np
import gymnasium as gym
import torch
from agent import Agent, Policy


algorithm = 'reinforce'   # which algorithm to use: 'reinforce' or 'actor_critic'
baseline = 20             # subtract this constant from the return to reduce gradient variance
run_name = f'baseline_{baseline}'  # just a label so I can tell different runs apart in the filename

NUM_EPISODES = 20000      # how many full episodes to train for
SEED = 42                 # random seed to ensure reproducibility
RENDER = False            # set to True to open a window and watch the agent train (slow!)


def train(algorithm, baseline, num_episodes, seed, checkpoint_dir, render=False):
    """Run the training loop and return (policy, ep_rewards, final_checkpoint_path)."""

    # Fix seeds so results are reproducible across runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)

    # Create the Hopper environment. render_mode='human' opens a visual window.
    # We skip rendering during training because it slows things down a lot.
    env = gym.make('Hopper-v4', render_mode='human' if render else None)

    # Build the neural network policy.
    # It reads how many inputs (state dimensions) and outputs (action dimensions) the env has.
    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])

    # The Agent wraps the policy and owns the optimizer and experience buffer.
    agent = Agent(policy)

    # Timestamp used in checkpoint filenames so I can tell runs apart.
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ep_rewards = []    # store total reward per episode so I can plot the "learning curve" later
    total_reward = 0.0 # running sum of all rewards across all episodes

    for i in range(num_episodes):
        done = False

        # Reset the environment at the start of each episode.
        # I vary the seed per episode so the agent sees slightly different starting states.
        state, _ = env.reset(seed=seed + i)
        ep_reward = 0.0

        # Inner loop: run one full episode
        while not done:
            # Ask the policy what action to take given the current state.
            # evaluation=False means we SAMPLE from the distribution (exploration).
            # action_log_prob is log π(a|s) – needed to compute the policy gradient.
            action, action_log_prob = agent.get_action(state, evaluation=False)

            # Send the action to the environment and get back the next state and reward.
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())

            # terminated = the robot fell; truncated = we hit the max step limit
            done = terminated or truncated
            ep_reward += reward

            # Store this transition so update_policy() can use it for the gradient update.
            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            # update_policy() is called every step, but the actual gradient update
            # only fires when done=True (end of episode) because REINFORCE is Monte Carlo:
            # it needs the full trajectory to compute the return G_t.
            agent.update_policy(algorithm=algorithm, baseline=baseline)

            state = next_state  # advance to the next state
        # End of episode

        ep_rewards.append(ep_reward)
        total_reward += ep_reward

        # Print a progress update every 100 episodes, including the moving average reward
        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}/{num_episodes}  avg-100: {np.mean(ep_rewards[-100:]):.1f}")


    env.close()

    # Save the final model. I encode total reward and mean reward in the filename
    # so I can compare runs just by looking at the filenames without loading them.
    final_path = os.path.join(checkpoint_dir, f"{algorithm}_{run_id}_{total_reward:.0f}_{num_episodes}_{total_reward/num_episodes:.1f}.pt")
    torch.save(policy.state_dict(), final_path)
    return policy, ep_rewards, final_path


def main():
    checkpoint_dir = f"part1/checkpoints/{algorithm}_{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(algorithm, baseline, NUM_EPISODES, SEED, checkpoint_dir, render=RENDER)


if __name__ == '__main__':
    main()