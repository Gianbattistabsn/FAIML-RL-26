# Task 1 – Guiding Questions

---

## Q1: What is the state space of the Hopper environment? Is it discrete or continuous?

The state space (also called observation space) is **continuous**.

At every timestep, the environment returns an observation that is a vector of **11 real-valued numbers**. Formally it is defined as:

$$Box(-\infty, +∞, shape=(11,), dtype=float64)$$

`Box` means it is a continuous multi-dimensional space where each element can take any real value within a given range (here unbounded). The 11 values describe the physical state of the Hopper's body:

| Index | Description |
|-------|-------------|
| 0 | Height of the torso (z position) |
| 1 | Angle of the torso (tilt) |
| 2 | Angle of the thigh joint |
| 3 | Angle of the leg joint |
| 4 | Angle of the foot joint |
| 5 | Forward velocity of the torso (x) |
| 6 | Vertical velocity of the torso (z) |
| 7 | Angular velocity of the torso |
| 8 | Angular velocity of the thigh joint |
| 9 | Angular velocity of the leg joint |
| 10 | Angular velocity of the foot joint |

The first 5 values are **positions/angles** (where is each part?), and the last 6 are **velocities** (how fast is each part moving?). Together they give the agent a complete picture of the robot's current configuration.

Note: the x-coordinate of the torso is excluded by default because knowing the absolute horizontal position would not help the agent learn a general hopping policy.

---

## Q2: What is the action space of the Hopper environment? Is it discrete or continuous?

The action space is also **continuous**.

At every timestep, the agent outputs a vector of **3 real-valued numbers**:

```
Box(-1.0, 1.0, shape=(3,), dtype=float32)
```

Each of the 3 values represents a **torque** applied to one of the Hopper's hinge joints:

| Index | Joint | Meaning |
|-------|-------|---------|
| 0 | Thigh joint | How hard to push the thigh |
| 1 | Leg joint | How hard to push the leg |
| 2 | Foot joint | How hard to push the foot |

Values range from -1 (maximum torque in one direction) to +1 (maximum torque in the other direction). This is different from a discrete action space where the agent would just pick from a fixed list of options (e.g., "jump", "stand still"). Here the agent can apply any precise combination of forces, which makes the problem harder but also more realistic.

---

## Q3: What is the mass of each link? What are source and target variants?

### What is a "link"?

In robotics and physics simulation, the body of a robot is modelled as a chain of **rigid segments** connected by **joints**. Each segment is called a **link** (or body). The Hopper has 4 links:

- **Torso** – the main upper body
- **Thigh** – the upper leg
- **Leg** – the lower leg
- **Foot** – the base that touches the ground

MuJoCo computes the mass of each link automatically from its geometry (shape + size), assuming a fixed material density. The default masses in `Hopper-v4` are:

| Link | Mass |
|------|------|
| Torso | ≈ 3.67 kg |
| Thigh | ≈ 4.06 kg |
| Leg | ≈ 2.78 kg |
| Foot | ≈ 5.32 kg |

### What are source and target variants?

This is a concept from **domain randomization** and **sim-to-real transfer**:

- The **source environment** is the environment the agent is *trained* in. It uses a fixed set of physical parameters (e.g., the default masses above).
- The **target environment** is a *different* version of the same environment where one or more physical parameters are changed (e.g., different link masses). This simulates a real robot that does not perfectly match the simulator, or simply a harder variant of the task.

The goal is to train the agent in the source environment and then test how well it performs in the target environment **without retraining**. If the agent still performs well despite the different physics, it means it has learned a robust, generalizable policy. This is important in robotics because simulators are never a perfect copy of the real world.

The specific mass values for the source and target variants used in this project are defined in the assignment document.
