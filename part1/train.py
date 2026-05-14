"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import datetime
import os
import random
import time

import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from agent import Agent, Policy


algorithm = 'reinforce'   # which algorithm to use: 'reinforce' or 'actor_critic'
baseline = 20             # subtract this constant from the return to reduce gradient variance
                          #   -1 -> adaptive: b = percentile_25(last-500 G_0), subtracted from every G_t
                          #    0 -> pure REINFORCE (optionally whiten G_t if normalize=True)
                          #   >0 -> fixed constant: subtracts baseline from every G_t
run_name = f'baseline_{baseline}'  # just a label so I can tell different runs apart in the filename

NUM_EPISODES = 20000      # how many full episodes to train for
SEED = 42                 # random seed to ensure reproducibility
RENDER = False            # set to True to open a window and watch the agent train (slow!)


def train(algorithm, baseline, num_episodes, seed, checkpoint_dir, render=False, normalize=False, device='auto', learning_rate=1e-3):
    """Run the training loop and return (policy, ep_rewards, final_checkpoint_path)."""

    # Fix seeds so results are reproducible across runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.xpu.manual_seed_all(seed)
    except AttributeError:
        pass  # Intel XPU not available on this machine
    np.random.seed(seed)
    random.seed(seed)

    # Create the Hopper environment. render_mode='human' opens a visual window.
    # We skip rendering during training because it slows things down a lot.
    env = gym.make('Hopper-v4', render_mode='human' if render else None)

    # Build the neural network policy.
    # It reads how many inputs (state dimensions) and outputs (action dimensions) the env has.
    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])

    # The Agent wraps the policy and owns the optimizer and experience buffer.
    agent = Agent(policy, device=device, learning_rate=learning_rate)

    # Timestamp used in checkpoint filenames so I can tell runs apart.
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ep_rewards = []     # store total reward per episode so I can plot the "learning curve" later
    ep_lengths = []     # store episode length (steps) per episode
    ep_actor_losses  = []     # actor loss value at the end of each episode
    ep_critic_losses  = []     # critic loss value at the end of each episode
    sigma_history = []  # σ snapshot (after softplus) every 100 episodes, for plotting
    total_reward = 0.0  # running sum of all rewards across all episodes
    best_avg_reward = float('-inf')  # track the best 100-episode moving average seen so far
    best_ckpt_path = None            # path to the current best checkpoint on disk
    train_start = time.time()        # wall-clock start for elapsed-time display
    interval_start = time.time()     # wall-clock start of the current 100-ep interval

    for i in range(num_episodes):
        done = False
        # Reset the environment at the start of each episode.
        # I vary the seed per episode so the agent sees slightly different starting states.
        state, _ = env.reset(seed=seed + i)
        ep_reward = 0.0
        ep_length = 0

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
            ep_length += 1

            # Store this transition so update_policy() can use it for the gradient update.
            agent.store_outcome(state, next_state, action_log_prob, reward, done)

            # update_policy() is called every step for actor_critic.
            # For REINFORCE, the gradient update fires after the full episode (see below).
            # All three baseline modes are handled inside update_policy; no resolution needed here.
            if algorithm != 'reinforce':
                actor_loss, critic_loss = agent.update_policy(algorithm=algorithm, baseline=baseline, normalize=normalize)

            state = next_state  # advance to the next state
        # End of episode
        if algorithm == 'reinforce':
            actor_loss, critic_loss = agent.update_policy(algorithm=algorithm, baseline=baseline, normalize=normalize)

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        if actor_loss is not None:
            ep_actor_losses.append(actor_loss)

        if actor_loss is not None:
            ep_critic_losses.append(critic_loss)
        
        total_reward += ep_reward

        # Every 100 episodes: print progress and check if this is the best model so far
        if (i + 1) % 100 == 0:
            recent_rewards = ep_rewards[-100:]
            avg_100   = np.mean(recent_rewards)
            min_100   = np.min(recent_rewards)
            max_100   = np.max(recent_rewards)
            avg_len   = np.mean(ep_lengths[-100:])
            avg_actor_loss  = np.mean(ep_actor_losses[-100:]) if ep_actor_losses else float('nan')
            avg_critic_loss  = np.mean(ep_critic_losses[-100:]) if ep_critic_losses else float('nan')
            sigma_eff = F.softplus(policy.sigma).detach().cpu().numpy()
            sigma_history.append((i + 1, sigma_eff.copy()))
            now = time.time()
            elapsed_total = now - train_start
            interval_secs = now - interval_start
            steps_in_interval = sum(ep_lengths[-100:])
            steps_per_sec = steps_in_interval / interval_secs if interval_secs > 0 else 0.0
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_total))
            interval_start = now  # reset for next 100-ep window
            print(
                f"Ep {i+1:>6}/{num_episodes} | "
                f"avg: {avg_100:7.1f}  min: {min_100:7.1f}  max: {max_100:7.1f} | "
                f"len: {avg_len:5.0f} | actor loss: {avg_actor_loss:8.4f} | critic loss: {avg_critic_loss:8.4f} | "
                f"σ: [{', '.join(f'{s:.3f}' for s in sigma_eff)}] | "
                f"best: {best_avg_reward:.1f} | "
                f"elapsed: {elapsed_str}  ({steps_per_sec:.0f} steps/s)"
            )

            if avg_100 > best_avg_reward:
                best_avg_reward = avg_100
                # Delete the previous best checkpoint to avoid filling disk
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                best_ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{algorithm}_best_ep{i+1}_avg{avg_100:.0f}.pt"
                )
                torch.save(policy.state_dict(), best_ckpt_path)
                print(f"  → New best! Saved to {os.path.basename(best_ckpt_path)}")


    env.close()

    # Save the final model. I encode total reward and mean reward in the filename
    # so I can compare runs just by looking at the filenames without loading them.
    final_path = os.path.join(checkpoint_dir, f"{algorithm}_{run_id}_{total_reward:.0f}_{num_episodes}_{total_reward/num_episodes:.1f}.pt")
    torch.save(policy.state_dict(), final_path)
    return policy, ep_rewards, ep_lengths, ep_actor_losses,ep_critic_losses, sigma_history, final_path


def main():
    checkpoint_dir = f"part1/checkpoints/{algorithm}_{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(algorithm, baseline, NUM_EPISODES, SEED, checkpoint_dir, render=RENDER)


if __name__ == '__main__':
    main()