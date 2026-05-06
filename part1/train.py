"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
from agent import Agent, Policy

algorithm = 'reinforce'
baseline = 20
NUM_EPISODES = 2000
SEED = 42
RENDER = True


def main():
    env = gym.make('Hopper-v4', render_mode='human' if RENDER else None)    
    policy = Policy(state_space=env.observation_space.shape[0], action_space = env.action_space.shape[0])
    
    # it automatically sets the device if available
    agent = Agent(policy)

    #TODO: implement training loop for REINFORCE and Actor-Critic using the agent defined in agent.py
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

        agent.update_policy(algorithm=algorithm, baseline=baseline)

if __name__ == '__main__':
    main()