import gymnasium as gym
import numpy as np



class RandomizationWrapper(gym.Wrapper):
    """
    Gym wrapper that applies domain randomization to the environment's physical parameters.

    This wrapper manages the randomization of environment properties, in particular mass,
    to facilitate Sim2Real transfer and improve policy robustness. It supports Uniform Domain
    Randomization (UDR) and Automatic Domain Randomization (ADR) strategies.

    Args:
        env (gym.Env): The base Gym environment to be wrapped.
        mode (str): The domain randomization strategy to use. Must be one of 'none' (disabled),
            'udr' (Uniform Domain Randomization), or 'adr' (Automatic Domain Randomization).
        mass_range (tuple[float, float]): The absolute minimum and maximum limits for mass
            randomization. The first value must be less than or equal to the second.
        seed (int, optional): Seed for the internal random number generator to ensure
            reproducibility.
        verbose (bool): If True, enables detailed logging of the randomization process.
        adr_delta (float): The step size used to expand or retract the randomization boundaries
            during ADR.
        adr_buffer_size (int): The number of boundary episodes to evaluate before triggering
            a boundary update check.
        adr_perf_low (float): The lower mean return threshold. If boundary performance drops
            below this value, the randomization range retracts.
        adr_perf_high (float): The upper mean return threshold. If boundary performance exceeds
            this value, the randomization range expands.
        adr_boundary_prob (float): The probability (between 0.0 and 1.0) of sampling at the
            exact boundaries instead of the interior range in ADR mode.

    Raises:
        ValueError: If an unrecognized `mode` is provided.
        ValueError: If `mass_range` is invalid (i.e., the minimum limit greater than the maximum).
        ValueError: If parameters `adr_delta`, `adr_buffer_size`, `adr_boundary_prob` fall outside
            valid numeric ranges.
        ValueError: If `adr_perf_low` is not lower than `adr_perf_high`
        """

    def __init__(self,
                 env: gym.Env,
                 mode: str = "none",
                 mass_range: tuple[float, float] = (1.0, 1.0),
                 seed: int = None,
                 verbose: bool = False,
                 *,  # adr specific from here, require full kwargs
                 adr_delta: float = 0.2,          # range increase size
                 adr_buffer_size: int = 20,       # performance buffer size for boundary performance evaluation
                 adr_perf_low: float = -25.0,     # mean return below this -> shrink range by delta
                 adr_perf_high: float = -10.0,    # mean return above this -> expand range by delta
                 adr_boundary_prob: float = 0.8   # must be in [0, 1], boundary sampling probability
                 ):

        super().__init__(env)

        # mode check
        if mode not in {"none", "udr", "adr"}:
            raise ValueError(f"Unknown mode={mode}; expected one of 'none', 'udr', 'adr'.")
        self.mode = mode

        # range check
        if mass_range[0] > mass_range[1]:
            raise ValueError(f"Invalid mass_range={mass_range}: lower bound must be less than or equal to upper bound.")
        self.mass_range = mass_range  # TODO! optional attribute, kept because present in original stub

        # global randomization limits TODO! deduced by stub nomenclature, ask if ADR can go outside the init range. otherwise the default range init is a dead end for adr.
        self._mass_min_limit, self._mass_max_limit = mass_range

        # environment parameters
        if mode == "adr":
            # adr starts from middle point of the range (can be changed, it is "tunable") !TODO ask for brainstorming
            self._mass_min = self._mass_max = (self._mass_min_limit + self._mass_max_limit) / 2
        else:
            # dynamic boundaries are set to the same as limits for udr logging
            self._mass_min = self._mass_min_limit
            self._mass_max = self._mass_max_limit

        # random number generator
        self._rng = np.random.default_rng(seed) # TODO!, this class' methods use a half open range [a,b), but in the ADR paper it specifies [a,b] interval for boundary expansion. Probably irrelevant, ask for confirmation.

        # verbose flag
        self.verbose = verbose

        # --- adr specific attributes ---
        # range increase size check
        if adr_delta <= 0.0:
            raise ValueError(f"Invalid adr_delta={adr_delta}; must be greater than 0.")
        self.adr_delta = adr_delta

        # performance buffer size check
        if adr_buffer_size < 1:
            raise ValueError(f"Invalid adr_buffer_size={adr_buffer_size}; must be greater or equal to 1.")
        self.adr_buffer_size = adr_buffer_size

        # performance thresholds check
        if adr_perf_low >= adr_perf_high:
            raise ValueError(
                f"Invalid performance thresholds adr_perf_low={adr_perf_low} and adr_perf_high={adr_perf_high};"
                " lower threshold must be less than upper threshold."
            )
        self.adr_perf_low = adr_perf_low
        self.adr_perf_high = adr_perf_high

        # boundary sampling probability check
        if not 0.0 <= adr_boundary_prob <= 1.0:
            raise ValueError(f"Invalid adr_boundary_prob={adr_boundary_prob}; must be between 0 and 1.")
        self.adr_boundary_prob = adr_boundary_prob

        # boundary sampling performance buffers
        self._buffer_low = []
        self._buffer_high = []

        # boundary flag
        self._current_boundary = None

        # total episode return
        self._episode_return = 0.0

        # Last sampled mass + a tag describing where it came from. Kept on the
        # wrapper so external callbacks (e.g. WandB) can read them directly
        # without parsing info dicts.
        # self.last_mass = None
        # one of: "none" | "udr" | "adr_low" | "adr_high" | "adr_interior"
        self._last_sample_type: str = "none"


    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_mass(self) -> float | None:
        """
        Sample the mass to apply for the next episode.

        This method determines the appropriate mass value to use in the simulation
        environment based on the active domain randomization strategy. Depending on
        the configured `mode`, it delegates the mass generation to either Uniform
        Domain Randomization (UDR) or Automatic Domain Randomization (ADR). If
        randomization is disabled (mode is "none"), the sampling process is bypassed.

        Side-effect:
            Updates `self._last_sample_type` to reflect the sampling method used.

        Returns:
            float | None: The sampled mass value based on the active randomization mode
            (UDR or ADR), or None if no randomization is selected.

        Raises:
            ValueError: If `self.mode` is unrecognized (not 'none', 'udr', or 'adr').
        """
        # no randomization mode
        if self.mode == "none":
            self._last_sample_type = "none"
            return None

        # Uniform Domain Randomization mode
        elif self.mode == "udr":
            return self._sample_mass_udr()

        # Automatic Domain Randomization mode
        elif self.mode == "adr":
            return self._sample_mass_adr()

        else: # unreachable
            raise ValueError(f"Unknown mode={self.mode}; expected one of 'none', 'udr', 'adr'.")

    def _sample_mass_udr(self) -> float:
        """
        Sample the mass using Uniform Domain Randomization (UDR).

        Generates a mass value sampled from a continuous uniform distribution
        bounded by `self._mass_min_limit` and `self._mass_max_limit`.

        Side-effect:
            Updates `self._last_sample_type` to "udr".

        Returns:
            float: A uniformly sampled mass value.
        """
        self._last_sample_type = "udr"
        return self._rng.uniform(self._mass_min_limit, self._mass_max_limit)

    def _sample_mass_adr(self) -> float:
        """
        Sample the mass using Automatic Domain Randomization (ADR).

        Generates a mass value dynamically based on performance boundaries. With a
        probability of `adr_boundary_prob`, the method performs boundary sampling
        (picking the exact lower or upper limit with equal probability). Otherwise,
        it performs interior sampling, drawing from a uniform distribution bounded
        by the current `_mass_min` and `_mass_max`.

        Side-effects:
            Updates `self._last_sample_type` to indicate "adr_low", "adr_high", or "adr_interior".
            Updates `self._current_boundary` to "low", "high", or None depending on the sampled location.

        Returns:
            float: A mass value sampled according to the current ADR bounds.
        """
        # boundary sampling with p = self.adr_boundary_prob
        if self._rng.random() < self.adr_boundary_prob:

            # upper or lower boundary sampling with equal probability
            if self._rng.random() < 0.5:

                # lower boundary sampling
                self._last_sample_type = "adr_low"
                self._current_boundary = "low"
                return self._mass_min

            else:

                # upper boundary sampling
                self._last_sample_type = "adr_high"
                self._current_boundary = "high"
                return self._mass_max

        # interior sampling with p = 1 - self.adr_boundary_prob
        else:
            self._last_sample_type = "adr_interior"
            self._current_boundary = None
            return self._rng.uniform(self._mass_min, self._mass_max)

    def _update_boundaries(self) -> None:
        """
        Evaluate boundary performance and adjust the randomization range.

        Acts when a boundary performance buffer reaches `adr_buffer_size`.
        - If mean reward > `adr_perf_high`: Range expands by `adr_delta` (outward).
        - If mean reward < `adr_perf_low`: Range shrinks by `adr_delta` (inward).

        Expansions are clamped by the global `mass_range` limits. Retractions are
        clamped to prevent the min boundary from exceeding the max boundary.

        Side-effects:
            Updates `self._mass_min` or `self._mass_max`.
            Clears the evaluated performance buffer (`_buffer_low` or `_buffer_high`).
            Prints update logs if `self.verbose` is True.
        """
        # lower boundary update
        if len(self._buffer_low) >= self.adr_buffer_size:
            # compute mean return over the buffer_low
            mean_return = np.mean(np.array(self._buffer_low))
            # expand lower boundary if mean_return is above the upper threshold
            if mean_return > self.adr_perf_high:
                self._mass_min = max(self._mass_min - self.adr_delta, self._mass_min_limit)  # bounded by initial range constraint TODO! ask if it was meant to be bounded or not
                # log boundary expansion update
                if self.verbose:
                    print(f"[adr] mass_min expanded to {self._mass_min:.3f} (mean_return={mean_return:.2f} > {self.adr_perf_high})")
            # retract lower boundary if mean_return is below the lower threshold
            elif mean_return < self.adr_perf_low:
                self._mass_min = min(self._mass_min + self.adr_delta, self._mass_max)  # bounded by self._mass_max
                # log boundary retraction update
                if self.verbose:
                    print(f"[adr] mass_min retracted to {self._mass_min:.3f} (mean_return={mean_return:.2f} < {self.adr_perf_low})")
            # dead zone: adr_perf_low <= mean_return <= adr_perf_high -> no change
            elif self.verbose:
                print(f"[adr] mass_min held at {self._mass_min:.3f} (mean_return={mean_return:.2f} in [{self.adr_perf_low}, {self.adr_perf_high}])")
            # reset buffer
            self._buffer_low = []

        # upper boundary update
        if len(self._buffer_high) >= self.adr_buffer_size:
            # compute mean return over the buffer_high
            mean_return = np.mean(np.array(self._buffer_high))
            # expand upper boundary if mean_return is above the upper threshold
            if mean_return > self.adr_perf_high:
                self._mass_max = min(self._mass_max + self.adr_delta, self._mass_max_limit)  # bounded by initial range constraint TODO! ask if it was meant to be bounded or not
                # log boundary expansion update
                if self.verbose:
                    print(f"[adr] mass_max expanded to {self._mass_max:.3f} (mean_return={mean_return:.2f} > {self.adr_perf_high})")
            # retract upper boundary if mean_return is below the lower threshold
            elif mean_return < self.adr_perf_low:
                self._mass_max = max(self._mass_max - self.adr_delta, self._mass_min)  # bounded by self._mass_min
                # log boundary retraction update
                if self.verbose:
                    print(f"[adr] mass_max retracted to {self._mass_max:.3f} (mean_return={mean_return:.2f} < {self.adr_perf_low})")
            # dead zone: adr_perf_low <= mean_return <= adr_perf_high -> no change
            elif self.verbose:
                print(f"[adr] mass_max held at {self._mass_max:.3f} (mean_return={mean_return:.2f} in [{self.adr_perf_low}, {self.adr_perf_high}])")
            # reset buffer
            self._buffer_high = []

    def _update_boundaries_dynamic(self) -> None:
        """
        Evaluate dynamic relative boundary updates (Placeholder).

        Currently uninstantiated. Intended to implement boundary expansions/retractions
        based on relative dynamic statistics (e.g., historical moving averages of returns)
        instead of hardcoded, static reward thresholds.
        """
        # TODO! idea: what if we did dynamic mean_return checking against a previous return or some statistic of previous returns, instead of hard reward thresholds? ask
        pass

    # -----------------------
    # Step
    # -----------------------

    def step(self, action):
        """
        Execute an environment step and track ADR performance.

        In ADR mode, this method accumulates rewards to calculate total episode
        return. If the episode ends and a boundary was being tested, the result
        is stored in the appropriate buffer for future boundary updates.

        Args:
            action: The action provided by the agent.

        Returns:
            tuple: (obs, reward, terminated, truncated, info) containing the
                standard Gymnasium step results.
        """
        # episode step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # update total episode return if adr
        if self.mode == "adr":
            self._episode_return += float(reward)

        # episode ended ?
        done = terminated or truncated

        # if episode ended and adr mode has done boundary sampling, update relative buffer
        if done and self.mode == "adr" and self._current_boundary is not None:  # TODO! do we consider truncated episodes valid contributions to boundary update?
            if self._current_boundary == "low":
                self._buffer_low.append(self._episode_return)
            else:  # "high"
                self._buffer_high.append(self._episode_return)
            self._update_boundaries()

            # TODO! see if info variable can be integrated with our stuff:
            # expose randomization state for downstream loggers (W&B etc.).
            # copy first so we don't mutate a dict the underlying env may reuse.
            # info = dict(info)
            # info["rand/mass"] = self._last_mass
            # info["rand/mass_min"] = self._mass_min
            # info["rand/mass_max"] = self._mass_max
            # info["rand/sample_type"] = self._last_sample_type

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):
        """
        Reset the environment and apply a new randomized mass.

        Resets the episode return accumulator to zero. Next, it samples a new mass
        value based on the active randomization strategy. If a valid mass is generated
        (UDR or ADR mode), the targeted object's mass is changed in the underlying
        physics simulation.

        Args:
            **kwargs: Additional keyword arguments passed directly to the base
                environment's reset method.

        Returns:
            tuple: the result of the call to the parent's reset method.
        """
        # reset episode reward before sampling
        self._episode_return = 0.0

        # sample new mass, returns None if mode is "none"
        new_mass = self._sample_mass()
        # save mass for logging in step
        # self.last_mass = new_mass

        # change object mass if UDR or ADR mode
        if new_mass is not None:
            # access and change object property
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]
            sim.physics_client.changeDynamics(bodyUniqueId=object_body_id,
                                              linkIndex=-1,
                                              mass=float(new_mass))
            # log change
            if self.verbose:
                print(f"[{self.mode}] mass={new_mass:.2f} "
                      f"range=[{self._mass_min:.2f},{self._mass_max:.2f}] "
                      f"type={self._last_sample_type}")
        # apply parent class reset
        return super().reset(**kwargs)

#print(f"[{self.mode}] mass={new_mass:.3f}")
#print(f"range=[{self.mass_min:.3f},{self.mass_max:.3f}]")
#print(f"type={self.last_sample_type}")

# notes: range and range limits are duplicated, could be normalized. possible to leave
# a public range attr and render the limits private and moving limits private