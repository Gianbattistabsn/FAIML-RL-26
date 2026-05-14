# Part 1 – REINFORCE on Hopper-v4

## What this part of the project does

This part of the project trains a robot to hop using **Reinforcement Learning (RL)**. The robot is Hopper-v4 from the Gymnasium library: a simulated 2D one-legged robot that has to learn to move forward without falling over.

The algorithm I implemented is called **REINFORCE** (also known as Monte Carlo Policy Gradient). It belongs to the family of *policy gradient* methods: instead of learning the value of each state (like Q-learning), we directly learn a *policy*: a function that maps a state to an action.

## Environment

| Property | Value |
|---|---|
| Environment | `Hopper-v4` (MuJoCo physics) |
| State space | 11 continuous values (joint angles + velocities) |
| Action space | 3 continuous torques, each in `[-1, 1]` |
| Episode end | Robot falls (`terminated`) or max steps reached (`truncated`) |

The agent never sees its absolute x-position: it only sees *how* it is moving (angles, velocities). Reward is given for moving forward and staying upright.

## Policy Architecture

The policy is a **neural network** that takes the current state as input and outputs an action. Because the action space is continuous (real-valued torques), the network does not output a single action directly: it outputs the **parameters of a probability distribution** over actions.

### Why a distribution instead of a single action?

In RL, we need the agent to *explore* during training: try different actions and see which ones lead to better outcomes. If the network always output the same deterministic action, it would never discover better strategies. By outputting a distribution, we can *sample* a slightly different action every time.

### The network structure

```
State (11 values)
        │
        ▼
  ┌────────────────┐
  │  Linear(11→64) │   fc1_actor
  │  + Tanh        │
  └────────────────┘
        │
        ▼
  ┌────────────────┐
  │  Linear(64→64) │   fc2_actor
  │  + Tanh        │
  └────────────────┘
        │
        ▼
  ┌────────────────┐
  │  Linear(64→3)  │   fc3_actor_mean
  └────────────────┘
        │
        ▼
   μ (mean action)        ← 3 values, one per joint

   σ (std dev)            ← 3 separate learned parameters (not from the MLP!)
        │
        ▼
   N(μ, σ)  per joint     ← 3 independent Gaussian distributions
```

- **Input**: 11-dimensional state vector $s \in \mathbb{R}^{11}$
- **Hidden layers**: two layers of 64 units each, with `Tanh` activation
- **Output**: 3 mean values $\mu \in \mathbb{R}^3$ (one per joint torque)
- **Sigma**: a separate vector $\sigma \in \mathbb{R}^3$ of learnable parameters, not connected to the MLP layers

### Tanh activation

`Tanh` squashes values to the range $(-1, 1)$. This helps gradients flow during training (avoids exploding or vanishing gradients) and keeps intermediate activations bounded.

## How exploration works: the learned $\sigma$

This is one of the most interesting design choices in the policy.

### The idea

At each step, the network predicts the **mean action** $\mu$ it thinks is best. But instead of always executing exactly $\mu$, it samples an actual action from a Gaussian distribution:

$$a \sim \mathcal{N}(\mu, \sigma)$$

- If $\sigma$ is **large** → the distribution is wide → the sampled action can be very different from $\mu$ → **more exploration**
- If $\sigma$ is **small** → the distribution is narrow → the sampled action is very close to $\mu$ → **more exploitation**

```
  Large σ (early training):            Small σ (late training):

       ___                                     |
      /   \                                    |
     /     \                               ____|____
    /       \                             /    |    \
───/─────────\───── actions          ────/─────|─────\─── actions
              μ                                μ
```

### What makes this special: $\sigma$ is *learned*

$\sigma$ is not a fixed hyperparameter: it is a `torch.nn.Parameter`, meaning the optimizer (Adam) also updates it during training through backpropagation:

```python
self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)
```

It starts at `init_sigma = 0.5` (reasonable initial exploration) and is updated alongside all the other network weights. If exploring leads to bad outcomes, the optimizer will push $\sigma$ down: the agent *decides by itself* how much to explore.

### Why softplus?

A standard deviation must always be positive. Raw parameters can go negative during optimization, so we apply $\text{softplus}$ before using $\sigma$:

$$\sigma_{\text{eff}} = \text{softplus}(\sigma) = \log(1 + e^{\sigma})$$

$\text{softplus}$ is a smooth approximation of ReLU that always outputs a positive value, and is differentiable everywhere (so gradients can flow back to $\sigma$).

```
softplus(x):

   3 │              ____─────
   2 │         ____/
   1 │    ____/
   0 │___/
     └──────────────────────── x
       -3  -2  -1   0   1   2
```

## Training vs Evaluation: two different behaviours

The `get_action()` method behaves differently depending on whether we are training or evaluating:

| Mode | What happens | Why |
|---|---|---|
| **Training** (`evaluation=False`) | Sample $a \sim \mathcal{N}(\mu, \sigma)$ | Need randomness to explore |
| **Evaluation** (`evaluation=True`) | Return $\mu$ directly | Want the best known action, no noise |

During evaluation, the agent is fully **deterministic**: it always picks the mean of the distribution. This makes the evaluation fair and reproducible.

## Why $\log \pi(a \mid s)$ and why the loss is a mean over steps

### Step 1 — the objective is over full trajectories

The objective $J(\theta)$ is the expected return of a full trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, \ldots)$:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0] = \int p_\theta(\tau) \cdot G_0(\tau) \, d\tau$$

The probability of a trajectory factorises over steps:

$$p_\theta(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \cdot p(s_{t+1} \mid s_t, a_t)$$

### Step 2 — the log-derivative trick

To compute $\nabla_\theta J$ we apply the identity $\nabla \log f = \frac{\nabla f}{f}$ to $p_\theta(\tau)$:

$$\nabla_\theta J = \mathbb{E}_\tau \left[ G_0 \cdot \nabla_\theta \log p_\theta(\tau) \right]$$

Only $\pi_\theta$ depends on $\theta$ (the transition model $p(s_{t+1}|s_t,a_t)$ and the initial state $p(s_0)$ do not). So:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

**The sum over $t$ appears because** $\log$ turns the product of per-step probabilities into a sum. Substituting:

$$\nabla_\theta J = \mathbb{E}_\tau \left[ G_0 \cdot \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

### Step 3 — causality: replace $G_0$ with $G_t$

Action $a_t$ cannot have caused rewards received **before** step $t$. Removing those past rewards does not bias the gradient but reduces its variance. The standard result is:

$$\nabla_\theta J = \mathbb{E}_\tau \left[ \sum_{t=0}^{T-1} G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

where $G_t = \sum_{k=0}^{T-1-t} \gamma^k r_{t+k}$ is the discounted return from step $t$ onwards.

### Step 4 — Monte Carlo estimate and the loss

We cannot compute the expectation over all possible trajectories. We approximate it with the single episode we just collected, treating each of the $T$ steps as an independent sample:

$$\nabla_\theta J \approx \sum_{t=0}^{T-1} G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) = T \cdot \frac{1}{T}\sum_{t=0}^{T-1} G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

We build a loss whose gradient equals this quantity (up to the $\frac{1}{T}$ scale, which does not affect direction):

$$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=0}^{T-1} \hat{G}_t \cdot \log \pi_\theta(a_t \mid s_t)$$

The minus sign turns gradient descent (PyTorch default) into gradient ascent on $J$.

### Dimensions in code

```python
# G_t     shape: (T,)  — one discounted return per step of the episode
# G_t     is then normalised: (G_t - mean) / (std + 1e-8)

# action_log_probs  shape: (T,)  — one scalar log π(a_t|s_t) per step
#   each entry = sum of log-probs over the 3 joints (already summed in get_action):
#   log π(a_t|s_t) = log π(a_t[0]|s_t) + log π(a_t[1]|s_t) + log π(a_t[2]|s_t)

actor_loss = (-(G_t - baseline) * action_log_probs).mean()
#              ↑ element-wise product, shape (T,)              ↑ mean over T steps
actor_loss.backward()
```

`G_t` is a plain tensor with no gradient — it comes from the environment. All the gradient flows through `action_log_probs`, which depends on $\mu$ and $\sigma$ through the Gaussian log-probability formula, so `.backward()` can reach all of $\theta$.

Why `.mean()` and not `.sum()`? Both give the same gradient direction, but `.sum()` scales with episode length $T$: a 500-step episode would produce a gradient 10× larger than a 50-step one, making the learning rate unreliable. `.mean()` keeps the gradient magnitude roughly constant regardless of episode length.

Because the 3 joints are modelled as **independent** Gaussians, the joint log-probability is a sum:

$$\log \pi(a \mid s) = \log \pi(a_0 \mid s) + \log \pi(a_1 \mid s) + \log \pi(a_2 \mid s)$$

```python
action_log_prob = normal_dist.log_prob(action).sum()  # sum over 3 joints → scalar
```

## How $\sigma$ gets updated: the gradient path

$\sigma$ is a `nn.Parameter`, so Adam will update it — but *how* does the gradient actually reach it?

### The full computation graph

Here is the complete picture of what has `requires_grad=True` and what does not:

```
θ (NN weights)                σ (Parameter)
    │                              │
    ▼                              │
  fc1 → tanh → fc2 → tanh → fc3    │
    │                              │
    ▼                              ▼
   μ (requires_grad=True)  softplus(σ) (requires_grad=True)
         │                       │
         └──────────┬────────────┘
                    ▼
            Normal(μ, σ)
                    │
                    ├──── .sample()  ─────► a   (requires_grad=FALSE)
                    │                          PyTorch uses torch.no_grad()
                    │                          internally — the randomness
                    │                          breaks the differentiable path
                    │
                    └──── .log_prob(a) ───► log π   (requires_grad=True)
                                │                because μ and σ appear
                                │                in the formula with a
                                │                treated as a CONSTANT
                                ▼
                    loss = -(G_t - baseline) · log π
                                │
                                ▼
                           .backward()
                                │
            ┌───────────────────┴────────────────────┐
            ▼                                        ▼
    ∂loss/∂μ → ∂μ/∂θ                        ∂loss/∂σ
    (updates all MLP weights)               (updates σ directly)
```

The key question is: **does `a` depend on $\mu$ and $\sigma$ in the graph?**

Mathematically yes — `a` was sampled from $\mathcal{N}(\mu, \sigma)$. But in PyTorch's graph: **no**. The sampled tensor `a` has `requires_grad=False`. It is a leaf with no history. When `log_prob(a)` is computed, `a` is just a fixed array of numbers — like writing `log_prob(torch.tensor([0.3, -0.1, 0.7]))`. The gradient flows only through $\mu$ and $\sigma$, not through `a`.

This is exactly why the log-derivative trick is necessary: it makes $\mu$ and $\sigma$ appear **explicitly** in the expression being differentiated, without needing to go through the non-differentiable sampling step.

### What does this gradient mean in practice?

For a Gaussian, the log-probability formula is:

$$\log \mathcal{N}(a;\, \mu, \sigma) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(a-\mu)^2}{2\sigma^2}$$

$\sigma$ appears explicitly, so its gradient is:

$$\frac{\partial \log \pi}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(a - \mu)^2}{\sigma^3}$$

- If the sampled action `a` landed **far from $\mu$** (i.e. $(a-\mu)^2$ is large) and the outcome was **good** ($G_t$ is high), the gradient pushes $\sigma$ **up** — the agent is rewarded for exploring, so it keeps exploring.
- If the sampled action landed far from $\mu$ and the outcome was **bad**, the gradient pushes $\sigma$ **down** — random exploration hurt, so the agent becomes more conservative.

Over time, $\sigma$ converges to a value that reflects how much uncertainty is still useful: once the policy is well-trained, $\sigma$ shrinks and the agent becomes nearly deterministic.

### Why not just pass gradients through the sample?

This is actually possible with a technique called the **reparameterization trick**, used in algorithms like SAC:

```python
# Reparameterized: a = mu + sigma * eps,  eps ~ N(0, 1)
# Now 'a' is differentiable w.r.t. both mu and sigma
```

REINFORCE does **not** use this trick — it uses `log_prob` instead, which is the original "score function estimator". Both approaches work, but the reparameterization trick generally has lower variance.

### To summarise: where does the gradient actually go?

The gradient from `.backward()` travels **two paths simultaneously** through `log_prob`:

```
loss
  │
  ▼
log π(a|s)  =  -½log(2πσ²)  -  (a - μ)² / 2σ²
                     │                  │
              ∂/∂σ                 ∂/∂μ
                     │                  │
                     ▼                  ▼
              σ (Parameter)            μ   ← output of the MLP
                                        │
                                        ▼
                               ∂μ/∂fc3 → ∂fc3/∂fc2 → ∂fc2/∂fc1 → ∂/∂θ
                               (chain rule back through the whole network)
```

- $\sigma$ is updated directly — it is a `Parameter` and appears explicitly in the formula
- $\theta$ (all MLP weights) are updated via chain rule through $\mu$ — because $\mu$ is the output of the network, which depends on $\theta$

What is **not** in this graph: `a`. Even though `a` was sampled from $\mathcal{N}(\mu, \sigma)$, once it is drawn it becomes a plain number with no gradient history. PyTorch never asks "how would `a` change if I tweaked $\theta$?" — it treats `a` as a fixed constant inside `log_prob`, and differentiates only w.r.t. $\mu$ and $\sigma$.

In one sentence: **the gradient updates both $\sigma$ and all of $\theta$, but it reaches $\theta$ through $\mu$ (the MLP output), not through the sampled action `a`**.

## Discounted Return $G_t$

REINFORCE is a **Monte Carlo** method: it collects a full episode, then assigns credit to each step. The return at step $t$ is:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

With $\gamma = 0.99$, a reward received 100 steps in the future is worth $0.99^{100} \approx 0.37$ times a reward received now. This makes near-future rewards more important.

We compute this efficiently by iterating backwards:

```python
for t in reversed(range(T)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
```

### Baseline subtraction

The loss is $-(G_t - \text{baseline}) \cdot \log \pi(a \mid s)$. Subtracting a constant $\text{baseline} = 20$ does **not** bias the gradient (the gradient of a constant is zero), but it **reduces variance**: if all returns are positive and large, the gradient signal is noisy. Centering the returns around zero makes updates more stable.

## The REINFORCE training loop

```
for each episode:
    │
    ├─ reset environment
    │
    └─ while not done:
          │
          ├─ state → Policy → N(μ, σ)
          ├─ sample action a ~ N(μ, σ)          ← exploration
          ├─ env.step(a) → next_state, reward
          ├─ store (s, s', log π(a|s), r, done)
          └─ if done:
                │
                ├─ compute G_t for all steps in episode
                ├─ loss = -(G_t - baseline) * log π(a|s)
                └─ optimizer.step()              ← update μ, σ and θ
```

The gradient update fires **once per episode** (at the very end), not at every step — this is what "Monte Carlo" means in this context.

## File structure

| File | Purpose |
|---|---|
| `part1/agent.py` | `Policy` (neural network) and `Agent` (optimizer + buffer + update logic) |
| `part1/train.py` | Training loop — exposes `train()` for import by the notebook |
| `part1/test.py` | Evaluation loop — exposes `evaluate()` for import by the notebook |
| `part1/colab_template/train_eval_reinforce_hopper.ipynb` | Notebook that calls `train()` and `evaluate()`, plots results, records video |
| `part1/checkpoints/` | Saved model weights (`.pt` files) |

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Hidden units | 64 | Per layer, both hidden layers |
| Activation | Tanh | After each hidden layer |
| Optimizer | Adam | lr = $10^{-3}$ |
| Discount factor $\gamma$ | 0.99 | |
| Baseline | 20 | Constant subtracted from $G_t$ |
| Initial $\sigma$ | 0.5 | Starting exploration level |
| Episodes (local) | 25 000 | In `train.py` standalone |
| Episodes (Colab) | 25 000 | In notebook |
| Seed | 42 | For reproducibility |

## What is still to do (Task 3)

The `Policy` class has a placeholder for a **critic network**:

```python
# TASK 3: critic network for actor-critic algorithm
```

The Actor-Critic algorithm adds a second neural network (the critic) that estimates the value $V(s)$ of each state. Instead of waiting for the full episode return $G_t$, the critic produces a *bootstrapped estimate* at each step:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \qquad \text{(TD error / advantage)}$$

This would significantly reduce variance compared to REINFORCE and allow the agent to update more frequently — one update per step instead of one per episode.
