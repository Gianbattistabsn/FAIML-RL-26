"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import numpy as np
import gymnasium as gym
import torch
from agent import Agent, Policy

# How many episodes to run during evaluation
NUM_EPISODES = 100
# Random seed – changing this gives different starting states
SEED = 42
# Set to True to watch the agent in a window, False for headless evaluation
RENDER = True


def evaluate(checkpoint_path, num_episodes=NUM_EPISODES, seed=SEED, render=True, video_dir=None):
    """Load a checkpoint and run evaluation episodes. Returns a list of episode rewards."""

    # If video_dir is given we wrap the env with RecordVideo to save .mp4 files.
    # This is useful in Colab where we can't open a window but still want to watch the agent.
    if video_dir is not None:
        from gymnasium.wrappers import RecordVideo
        # rgb_array mode gives us frames as numpy arrays, which RecordVideo needs to write video
        env = gym.make('Hopper-v4', render_mode='rgb_array')
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True, name_prefix="eval")
    else:
        # Running locally: open a window if RENDER=True, otherwise run silently
        env = gym.make('Hopper-v4', render_mode='human' if render else None)

    # Recreate the same network architecture that was used during training,
    # then load the saved weights into it.
    policy = Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
    policy.load_state_dict(torch.load(checkpoint_path, weights_only=False))

    # Switch to eval mode – disables things like dropout that are only needed during training
    policy.eval()
    agent = Agent(policy)

    eval_rewards = []
    for i in range(num_episodes):
        done = False
        # Use a different seed each episode so we test on varied starting conditions
        state, _ = env.reset(seed=seed + i)
        ep_reward = 0.0

        while not done:
            # evaluation=True means we take the MEAN of the Gaussian (no sampling).
            # This makes the agent deterministic and removes noise from the evaluation.
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            ep_reward += reward
            done = terminated or truncated

        eval_rewards.append(ep_reward)

    # close() also finalises the video files if RecordVideo is active
    env.close()
    return eval_rewards


def main():
    import os
    # train.py main() saves into a subdirectory: part1/checkpoints/{algorithm}_{run_name}/
    # We scan all subdirectories to find the most recent checkpoint.
    checkpoint_root = "part1/checkpoints"
    ckpts = sorted([
        os.path.join(dirpath, f)
        for dirpath, _, files in os.walk(checkpoint_root)
        for f in files if f.endswith('.pt')
    ])
    if not ckpts:
        print(f"No checkpoints found under {checkpoint_root}/")
        return
    print(ckpts)

    ckpt = ckpts[-1]
    print(f"Loading: {ckpt}")
    rewards = evaluate(ckpt, render=RENDER)
    print(f"Mean reward: {np.mean(rewards):.2f}")


if __name__ == '__main__':
    main()