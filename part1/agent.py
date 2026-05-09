import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    """Compute discounted returns G_t for each timestep t in an episode.

    REINFORCE needs the *return* G_t, not just the immediate reward r_t.
    The return is the sum of all future rewards, each one discounted by gamma:

        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    We iterate backwards so we can reuse the value computed for t+1:

        G_t = r_t + gamma * G_{t+1}

    gamma < 1 (e.g. 0.99) makes rewards far in the future worth less,
    which keeps G_t finite and prioritises near-term outcomes.
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time.
        # sigma controls how "spread out" the action distribution is:
        #   - large sigma  -> wide distribution -> more exploration (random actions)
        #   - small sigma  -> narrow distribution -> more exploitation (near-deterministic)
        # We use softplus (a smooth version of ReLU) to guarantee sigma > 0,
        # because a standard deviation must always be positive.
        # sigma is a learnable parameter: the network decides how much to explore.
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma) #self.sigma = [0.5,0.5,0.5]


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1) #V(s) is a scalar function



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x, critic=False): 
        """
            Actor
        """
        if(not critic):

            # Pass the state through the 3-layer MLP to get the mean action.
            # Tanh squashes activations to (-1, 1), which helps training stability.
            x_actor = self.tanh(self.fc1_actor(x))
            x_actor = self.tanh(self.fc2_actor(x_actor))
            action_mean = self.fc3_actor_mean(x_actor)   # shape: (action_space,)

            # Apply softplus to sigma so it is always positive (required for a std dev).
            sigma = self.sigma_activation(self.sigma)    # shape: (action_space,)

            # Build a Gaussian distribution N(action_mean, sigma) over the action space.
            # Hopper has 3 continuous joints, so this is a 3-dimensional Gaussian
            # (one independent Normal per joint).
            normal_dist = Normal(action_mean, sigma)
            return normal_dist

        """
            Critic
        """
        
        if(critic):

            # TASK 3: forward in the critic network
            # for now the critic NN has the same structure of the actor. The only thing changed
            # is that the last layer doesn't have an activation. This is because V(s) can go from
            # +Inf to -Inf
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))
            value_func = self.fc3_critic(x_critic)
            return value_func


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device=device
        if torch.cuda.is_available():
            self.train_device = 'cuda'
        elif torch.xpu.is_available():
            self.train_device = 'xpu'
        else:
            self.train_device = 'cpu'
        
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self, algorithm='reinforce', baseline = 0):
        # the list of states, actions etc... is moved to local variables. The global variables are reset
        # at each update
        
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)


        
        # TASK 2:
        #   - compute discounted returns
        G_t = discount_rewards(rewards, self.gamma)
        #   - compute policy gradient loss function given actions and returns
        if algorithm == 'reinforce' and done[-1] == True:
            if baseline == 0:
                # Whitening: normalise G_t to zero mean, unit std within the episode.
                # Makes gradient scale consistent regardless of absolute reward magnitude.
                G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-8)
            actor_loss = (-(G_t - baseline) * action_log_probs).mean()
            # The minus sign turns gradient descent into gradient ascent,
            # since we want to MAXIMISE expected return.

            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()
            self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []
            return actor_loss.item(), actor_loss.item()
        
        elif algorithm=='actor_critic':
            #computing bootstrapped estimate, aka R_t+1 + gamma V(S_t+1)
            current_state = states[-1]
            next_state = next_states[-1]
            reward = rewards[-1]
            if done[-1] == False: 
                next_value = self.policy(next_state, critic = True) #calculate V(S_t+1)
                estimate_value = reward + self.gamma*next_value
            else:
                # if we are in the terminal state we should't be able to compute V(S_t+1)
                # Therefore we assume it to be 0
                estimate_value = reward

            #compute the advantage δ_t
            value = self.policy(current_state, critic = True) #V(S_t)
            delta = estimate_value.detach() - value

            #compute actor and critic loss
            current_action_log_prob = action_log_probs
            delta_detached = delta.detach()
            actor_loss = -delta_detached* current_action_log_prob 

            critic_loss = 0.5 * delta**2

            #Saving the current values of the losses, needed for the train analysis
            actor_loss_value = actor_loss.item()
            critic_loss_value = critic_loss.item()

            #compute the gradients
            self.optimizer.zero_grad() #puts to 0 the gradients. If not used it would be grad = old_grad+new_grad

            critic_loss.backward()

            #step the gradient
            self.optimizer.step()

            self.optimizer.zero_grad() 

            actor_loss.backward()

            self.optimizer.step()

            #reset the lists after each iteration of the episode
            self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

            return actor_loss_value, critic_loss_value
        
        return None, None

    def get_action(self, state, evaluation=False):
        """Given the current state, return an action and its log-probability.

        During training  (evaluation=False): sample a random action from the
        policy distribution so the agent can explore.
        During evaluation (evaluation=True):  return the distribution mean,
        which is the single best action the policy knows (no randomness).
        """
        # Convert the numpy state array coming from the env into a PyTorch tensor
        # on the correct device (CPU or GPU).
        x = torch.from_numpy(state).float().to(self.train_device)

        # Forward pass: the network returns a Normal distribution N(mu, sigma)
        # parameterised by the current policy weights.
        normal_dist = self.policy(x)

        if evaluation:  # deterministic: just use the mean of the distribution
            return normal_dist.mean, None

        else:   # stochastic: draw one sample from N(mu, sigma)
            action = normal_dist.sample()   # shape: (action_space,) = (3,)

            # Why do we need log π(a|s)? 
            # The REINFORCE gradient is:  ∇J = E[ G_t · ∇ log π(a_t|s_t) ]
            # We need log π(a|s) so PyTorch can differentiate through it
            # with .backward() later.
            #
            # Why sum() over the 3 action dimensions? 
            # The 3 joints are modelled as *independent* Gaussians, so the
            # joint probability of taking all 3 actions together is:
            #
            #  p(a) = p(a[0]) * p(a[1]) * p(a[2])
            #
            # Taking the log turns the product into a sum (log of a product
            # equals the sum of logs):
            #
            #   log p(a) = log p(a[0]) + log p(a[1]) + log p(a[2])
            #
            # normal_dist.log_prob(action) gives [log p(a[0]), log p(a[1]), log p(a[2])],
            # and .sum() collapses them into a single scalar log π(a|s).
            action_log_prob = normal_dist.log_prob(action).sum()
            #print(action_log_prob)

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        """Buffer one (s, s', log π(a|s), r, done) transition.

        REINFORCE is a Monte Carlo method: it needs the *complete* episode
        before it can compute G_t and update the policy.  We therefore store
        every transition and only call update_policy() at the episode end.
        """
        # Store states as plain CPU tensors (no device move yet) to avoid
        # unnecessary CPU<->GPU transfers; the move happens in update_policy().
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)  # keeps the computation graph alive
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

