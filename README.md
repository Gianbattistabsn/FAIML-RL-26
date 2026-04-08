# Starting code for course project of FAIML - 01VSDWS

Official assignment at [Google Doc](https://docs.google.com/document/d/1AXgLXux3l69vDAPLL-UYD3luFOw3JbyR-pLCS2yuNZk/edit?usp=sharing)



## Getting started

Before starting to implement your own code, make sure to:
1. read and study the material provided (see Section 1 of the assignment)
2. read the documentation of the main packages you will be using ([Gymnasium](https://gymnasium.farama.org), [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))
3. play around with the code in the template to familiarize with all the tools. Especially with the `test_random_policy.py` script.


### 1. Local

if you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening.
We highly suggsest
**Dependencies**
- Run `pip install -r requirements.txt`

Check your installation by launching `python test_random_policy.py`.


### 2. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/)

- Download all files contained in the `colab_template` folder in this repo (inside phase_1 folder).
- Load the `test_random_policy.ipynb` file on [https://colab.research.google.com/](colab) and follow the instructions on it.

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once, to best understand what is going on with the underlying Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.

## Project structure

```
FAIML-RL-26/
├── README.md
├── phase1/ <-- about Hopper
│   ├── agent.py
│   ├── requirements.txt
│   ├── test_random_policy.py
│   ├── train.py
│   └── colab_template/
│       └── test_random_policy.ipynb
└── phase2/ <-- about PushTask
    ├── eval_sb3.py
    ├── rand_wrapper.py <-- randomization wrapper for UDR/ADR
    ├── train_sb3.py
    └── panda-gym/
        └── panda_gym/ (main package)
            └── envs/
                ├── core.py
                ├── panda_tasks.py
                ├── robots/
                │   └── panda.py
                └── tasks/
                    ├── flip.py
                    ├── pick_and_place.py
                    ├── push.py <-- you will use this environment
                    ├── reach.py
                    ├── slide.py
                    └── stack.py
        
```