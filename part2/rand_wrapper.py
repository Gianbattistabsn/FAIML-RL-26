import gymnasium as gym
import numpy as np


class RandomizationWrapper(gym.Wrapper):
    """
    Gym wrapper that applies domain randomization to the environment's physical parameters.

    This wrapper manages the randomization of environment properties (currently focused on mass)
    to facilitate Sim2Real transfer or improve policy robustness. It supports Uniform Domain
    Randomization (UDR) and Automatic Domain Randomization (ADR) strategies. The wrapper
    initializes the boundaries for randomization and sets up an isolated random number generator.

    Args:
        env (gym.Env): The base Gym environment to be wrapped.
        mass_range (tuple[float, float]): The absolute minimum and maximum limits for mass
            randomization. The first value must be strictly less than the second.
            Defaults to (1.0, 1.0).
        mode (str): The domain randomization strategy to use. Must be one of 'none' (disabled),
            'udr' (Uniform Domain Randomization), or 'adr' (Automatic Domain Randomization).
            Defaults to "none".
        seed (int, optional): Seed for the internal random number generator to ensure
            reproducibility. Defaults to None.
        verbose (bool): If True, enables detailed logging of the randomization process.
            Defaults to False.

    Raises:
        ValueError: If an unrecognized `mode` is provided.
        ValueError: If `mass_range` is invalid (i.e., the minimum limit greater than the maximum).
    """
    def __init__(self,
                 env: gym.Env,
                 mass_range: tuple[float, float] = (1.0, 1.0),
                 mode: str = "none",
                 seed: int = None,
                 verbose: bool = False):

        super().__init__(env)

        # mode check
        if mode not in {"none", "udr", "adr"}:
            raise ValueError(f"Unknown mode={mode}; expected one of 'none', 'udr', 'adr'.")
        self.mode = mode

        # range check
        if mass_range[0] > mass_range[1]:
            raise ValueError(f"Invalid mass range: {mass_range}; beginning must be less than ending.")
        self.mass_range = mass_range
        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range

        # random number generator
        self._rng = np.random.default_rng(seed)
        # verbose flag
        self.verbose = verbose

        # attributes init
        self.last_sample_type = "none"
        self.mass_min = None
        self.mass_max = None


    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self):
        """
        Sample the mass to apply for the next episode.

        This method determines the appropriate mass value to use in the simulation
        environment based on the active domain randomization strategy. Depending on
        the configured `mode`, it delegates the mass generation to either Uniform
        Domain Randomization (UDR) or Automatic Domain Randomization (ADR). If
        randomization is disabled (mode is "none"), the sampling process is bypassed.

        Side-effect:
            Updates `self.last_sample_type` to reflect the sampling method used.

        :return: The sampled mass value based on the active randomization mode
                 (UDR or ADR), or None if no randomization is configured.
        :raises ValueError: If `self.mode` is unrecognized (not 'none', 'udr', or 'adr').
        """

        # no randomization mode
        if self.mode == "none":
            self.last_sample_type = "none"
            return None

        # Uniform Domain Randomization mode
        elif self.mode == "udr":
            return self._sample_mass_udr()

        # Automatic Domain Randomization mode
        elif self.mode == "adr":
            return self._sample_mass_adr()

        else: # unreachable if all ok
            raise ValueError(f"Unknown mode={self.mode}; expected one of 'none', 'udr', 'adr'.")

    def _sample_mass_udr(self):
        """
        Sample the mass using Uniform Domain Randomization (UDR).

        Generates a mass value sampled from a continuous uniform distribution
        bounded by `self.mass_min_limit` and `self.mass_max_limit`.

        Side-effect:
            Updates `self.last_sample_type` to "udr".

        :return: A uniformly sampled mass value as a float.
        """
        self.last_sample_type = "udr"
        return float(self._rng.uniform(self.mass_min_limit, self.mass_max_limit))

    def _sample_mass_adr(self):
        """
        Sample the mass using Automatic Domain Randomization (ADR).

        Generates a mass value ...

        Side-effect:
            Updates `self.last_sample_type` to "adr".

        :return: ...
        """
        self.last_sample_type = "adr"
        raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        # Optionally, you can add here extra logic

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):

        new_mass = self._sample_mass()

        if new_mass is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(bodyUniqueId=object_body_id,
                                              linkIndex=-1,
                                              mass=float(new_mass))
            # print info
            if self.verbose:
                print(f"[{self.mode}] mass={new_mass:.2f} "
                      f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                      f"type={self.last_sample_type}")

        return super().reset(**kwargs)
