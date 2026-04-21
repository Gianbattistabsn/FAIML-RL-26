"""Test a random policy on the Gym Hopper environment

    This script is your "hello world" for Reinforcement Learning.
    It creates a Hopper robot (a one-legged figure that tries to hop forward)
    and makes it act completely at random — no learning at all.

    The goal here is just to get familiar with how the RL loop works:
      - The ENVIRONMENT is the simulator (the Hopper world).
      - The AGENT is whoever decides what action to take (here: pure random).
      - At each timestep the agent picks an ACTION, the environment responds
        with a new STATE and a REWARD, and the process repeats.

    Play around with this code to get familiar with the Hopper environment.

    For example:
      - What happens if you don't reset the environment even after the episode is over?
      - When exactly is the episode over?
      - What is an action here?
"""

import gymnasium as gym


def main():
    """
    RENDERING
    If render=True, a window opens so you can watch the Hopper move.
    If render=False, everything still runs, just without the visual window
    (could be useful when training on a server with no display).
    """
    
    render = True

    if render:
        env = gym.make('Hopper-v4', render_mode='human') # open a visual window
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array') # no visual window

    model = env.unwrapped.model
    for i in range(model.nbody):
        print(model.body(i).name, model.body(i).mass)

    """
    STATE SPACE  (also called "observation space")
    The STATE is all the information the agent can observe about the world
    at a given moment. For Hopper it is a vector of 11 numbers describing
    angles, velocities, etc. of the robot's joints.
    """ 

    """
    ACTION SPACE
    The ACTION is what the agent decides to do. For Hopper it is a vector
    of 3 continuous numbers representing torques applied to the joints.
    Each value is in [-1, 1].
    """
    
    print('State space:', env.observation_space)   # what the agent can see
    print('Action space:', env.action_space)       # what the agent can do

    """
    EPISODES
    An EPISODE is one complete "game" from start to finish.
    It ends either when:
        - the robot falls over  -> "terminated" = True
        - a maximum number of steps is reached -> "truncated" = True
    """
    n_episodes = 50
    
    for ep in range(n_episodes):
        done = False
        # env.reset() puts the robot back to its starting position and returns
        # the initial state (observation) plus some extra info (usually empty).
        state, info = env.reset() # it changes a bit run by run.
        print(f'\nEpisode {ep} started. Initial state: {state}, info: {info}\n')
        while not done:   # keep acting until the episode ends
            """
            RANDOM POLICY
            env.action_space.sample() picks a completely random action.
            A real RL agent would choose actions based on what it has learned
            so far but here we just act randomly as a baseline.
            """
            action = env.action_space.sample()

            """
            env.step(action) applies the action to the simulator for one timestep.
            It returns:
                state: the new observation after the action
                reward: a scalar score (for Hopper: roughly = forward speed)
                terminated: True if the robot fell / reached a terminal state
                truncated: True if the episode hit the max-step time limit
                _: extra debug info (we ignore it here)
            """
            state, reward, terminated, truncated, _ = env.step(action)

            # The episode is over if EITHER termination condition is met.
            done = terminated or truncated

            if render:
                env.render()   # refresh the visual window
                print(f'Action: {action}, State: {state}, Reward: {reward}')

        print() 


if __name__ == '__main__':
    main()