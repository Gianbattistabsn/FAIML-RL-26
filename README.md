
# Robust Reinforcement Learning for Control and Robotics

This repository contains the Reinforcement Learning project developed for the
**Fundamentals of Artificial Intelligence, Machine and Deep Learning** course at
Politecnico di Torino.

> **For a complete description of the methodology, experimental setup, results
> and conclusions, please refer to the
> [Project Report](./FAIML_RL_Project_Report.pdf).**
>
> The report is the main reference for evaluating the project, while this
> repository contains the corresponding implementations and experimental code.

## Project Overview

The project investigates reinforcement learning techniques for continuous-control
and robotic-manipulation tasks. It is divided into two parts.

### Part 1 — Continuous Control

We implemented and evaluated policy-gradient algorithms for the
**MuJoCo Hopper** environment.

The main activities included:

- implementing **REINFORCE** from scratch in PyTorch;
- implementing an **Actor–Critic** agent;
- analysing training stability, convergence and policy performance;
- comparing different architectural and hyperparameter choices.

### Part 2 — Robust Robotic Manipulation

We trained reinforcement learning agents for a robotic pushing task based on the
**Franka Panda** robot.

The main activities included:

- training and comparing **PPO** and **SAC** agents;
- introducing variations in object dynamics and environment parameters;
- evaluating agent robustness under distribution shifts;
- comparing **Uniform Domain Randomization (UDR)**,
  **Automatic Domain Randomization (ADR)** and
  **Bounded Automatic Domain Randomization (BADR)**.

The objective was to study how domain-randomization strategies can improve the
generalization and robustness of reinforcement learning policies when the
deployment environment differs from the training environment.


## Authors

Project developed by:

* Gianbattista Busonera
* Giovanni Casati
* Alessandro de Stasi
* Gabriele Tebano

## Course

**Fundamentals of Artificial Intelligence, Machine and Deep Learning**
Politecnico di Torino — Academic Year 2025/2026

---

The original repository was provided as the starting template for the course
assignment. The implementations, experiments, analyses and project report
contained in this fork were developed by the project team.

```
```
