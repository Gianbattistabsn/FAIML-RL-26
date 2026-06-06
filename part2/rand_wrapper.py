import gymnasium as gym
import numpy as np


class RandomizationWrapper(gym.Wrapper):
    """
    Gym wrapper that applies domain randomization to the environment's physical parameters.

    This wrapper manages the randomization of environment properties, in particular mass,
    to facilitate Sim2Real transfer and improve policy robustness. It supports Uniform Domain
    Randomization (UDR), canonical Automatic Domain Randomization (ADR), and Bounded ADR
    (BADR) strategies.
    """

    # Physical-plausibility floor for the object mass. In canonical ADR mode the lower
    # boundary is clamped against this floor on expansion, to prevent meaningless
    # non-positive mass. BADR mode uses the user-supplied `mass_range[0]` as its lower
    # expansion floor instead, so this constant is unused there.
    _MIN_PHYSICAL_MASS: float = 1e-3

    def __init__(self,
                 env: gym.Env,
                 mode: str = "none",
                 mass_range: tuple[float, float] = (1.0, 1.0),
                 seed: int = None,
                 *,  # adr specific from here, require full kwargs
                 adr_delta: float = 0.2,          # range increase size
                 adr_buffer_size: int = 20,       # performance buffer size for boundary performance evaluation
                 adr_perf_low: float = -25.0,     # mean return below this -> shrink range by delta
                 adr_perf_high: float = -10.0,    # mean return above this -> expand range by delta
                 adr_boundary_prob: float = 0.8   # must be in [0, 1], boundary sampling probability
                 ):

        super().__init__(env)

        # mode check
        if mode not in {"none", "udr", "adr", "badr"}:
            raise ValueError(f"Unknown mode={mode}; expected one of 'none', 'udr', 'adr', 'badr'.")
        self.mode = mode

        # range check
        if mass_range[0] > mass_range[1] or mass_range[0] < self._MIN_PHYSICAL_MASS:
            raise ValueError(
                f"Invalid mass_range={mass_range}: lower bound must be less than or equal to upper bound,"
                f" and greater than or equal to {self._MIN_PHYSICAL_MASS}."
            )

        # outer hard limits used by BADR (ignored by 'none', 'udr', 'adr')
        self._mass_min_limit, self._mass_max_limit = mass_range

        if mode == "adr" or mode == "badr":
            # ADR / BADR: collapse the sampling range to the midpoint; boundaries grow from there
            self._mass_min = self._mass_max = (mass_range[0] + mass_range[1]) / 2
        else:
            self._mass_min, self._mass_max = mass_range

        # random number generator
        self._rng = np.random.default_rng(seed)

        # --- adr / badr specific attributes ---
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

        # one of: "none" | "udr" | "adr_low" | "adr_high" | "adr_interior"
        self._last_sample_type: str = "none"

        # BADR per-side saturation flags. Flip to True (via `_expand_bound`) when the
        # corresponding bound reaches its user-provided hard limit. Once True, the side
        # is locked: no further boundary sampling, no expansion, no retraction.
        self._badr_upper_saturated = False
        self._badr_lower_saturated = False

    # attribute getters for wandb logging
    @property
    def mass_min(self) -> float:
        return self._mass_min

    @property
    def mass_max(self) -> float:
        return self._mass_max

    @property
    def last_sample_type(self) -> str:
        return self._last_sample_type

    # -----------------------
    # Mass Sampling
    # -----------------------
    def _sample_mass(self) -> float | None:
        """
        Sample the mass to apply for the next episode.

        This method determines the appropriate mass value to use in the simulation
        environment based on the active domain randomization strategy. Depending on
        the configured `mode`, it delegates the mass generation to UDR (`_sample_mass_udr`)
        or to the shared ADR/BADR sampler (`_sample_mass_adr`). The difference between
        canonical ADR and BADR only matters when boundaries are updated, not when a
        sample is drawn from the current range — both modes share the sampler. If
        randomization is disabled (mode is "none"), the sampling process is bypassed.
        """
        # no randomization mode
        if self.mode == "none":
            self._last_sample_type = "none"
            return None

        # Uniform Domain Randomization mode
        elif self.mode == "udr":
            return self._sample_mass_udr()

        # Automatic Domain Randomization mode, canonical or bounded
        elif self.mode == "adr" or self.mode == "badr":
            return self._sample_mass_adr()

        else: # unreachable
            raise ValueError(f"Unknown mode={self.mode}; expected one of 'none', 'udr', 'adr', 'badr'.")

    def _sample_mass_udr(self) -> float:
        """
        Sample the mass using Uniform Domain Randomization (UDR).

        Generates a mass value sampled from a continuous uniform distribution
        bounded by `self._mass_min` and `self._mass_max`. In UDR mode these bounds
        are never updated by the wrapper (`_update_boundaries` only runs in ADR /
        BADR mode), so they remain equal to the `mass_range` passed at construction
        time for the entire training run.
        """
        self._last_sample_type = "udr"
        return self._rng.uniform(self._mass_min, self._mass_max)

    def _sample_mass_adr(self) -> float:
        """
        Sample the mass using Automatic Domain Randomization.

        Shared by both `adr` and `badr` modes as the two strategies differ only in how
        boundaries are updated (see `_expand_bound`), not in how a mass is drawn from
        the current range. In ADR mode the saturation flags never flip, so the
        symmetric "boundary with prob p, interior with prob 1-p" behavior always
        applies. In BADR mode the sampling behavior degrades as sides saturate
        against their hard limits.
        """
        # boundary sampling with p = self.adr_boundary_prob
        if (not (self._badr_upper_saturated and self._badr_lower_saturated)
                and self._rng.random() < self.adr_boundary_prob):

            can_low = not self._badr_lower_saturated
            can_high = not self._badr_upper_saturated

            # coin flip if both sides available, else take the remaining one
            pick_low = (self._rng.random() < 0.5) if (can_low and can_high) else can_low

            if pick_low:
                self._last_sample_type = "adr_low"
                self._current_boundary = "low"
                return self._mass_min
            else:
                self._last_sample_type = "adr_high"
                self._current_boundary = "high"
                return self._mass_max


        # interior sampling with p = 1 - self.adr_boundary_prob
        # or p = 1 if saturated
        self._last_sample_type = "adr_interior"
        self._current_boundary = None
        return self._rng.uniform(self._mass_min, self._mass_max)

    def _update_boundaries(self) -> None:
        """
        Evaluate boundary performance and adjust the randomization range.

        Acts when a boundary performance buffer reaches `adr_buffer_size`.
        - If mean reward > `adr_perf_high`: range expands by `adr_delta` (outward).
        - If mean reward < `adr_perf_low`: range shrinks by `adr_delta` (inward).
        - adr_perf_low <= mean_return <= adr_perf_high  -> dead zone, no update.

        Expansion clamping is mode-dependent (see `_expand_bound`):
            - In ADR mode the lower boundary is floored at `_MIN_PHYSICAL_MASS` and the
              upper boundary is unbounded (canonical ADR behavior).
            - In BADR mode both boundaries are additionally clamped against
              `_mass_min_limit` / `_mass_max_limit` so the range never exceeds the
              user-provided `mass_range`. When either side touches its hard limit it
              is marked saturated by `_expand_bound`; subsequent expansion or
              retraction attempts against the saturated side are silently dropped.
              When both sides are saturated this method returns immediately without
              processing either buffer.

        Retractions are clamped against the opposite dynamic bound so that
        `_mass_min <= _mass_max` always holds (mode-agnostic), with the additional
        BADR rule that saturated sides are not retracted.
        """
        if self._badr_upper_saturated and self._badr_lower_saturated:
            return
        # lower boundary update if buffer size reached
        if len(self._buffer_low) >= self.adr_buffer_size:

            # compute mean return over the buffer_low
            mean_return = np.mean(np.array(self._buffer_low))

            # expand lower boundary if mean_return is above the upper threshold
            if mean_return > self.adr_perf_high:
                self._expand_bound("low")

            # retract lower boundary if mean_return is below the lower threshold
            elif mean_return < self.adr_perf_low:
                self._retract_bound("low")

            # reset buffer
            self._buffer_low = []

        # upper boundary update
        if len(self._buffer_high) >= self.adr_buffer_size:

            # compute mean return over the buffer_high
            mean_return = np.mean(np.array(self._buffer_high))

            # expand upper boundary if mean_return is above the upper threshold
            if mean_return > self.adr_perf_high:
                self._expand_bound("high")

            # retract upper boundary if mean_return is below the lower threshold
            elif mean_return < self.adr_perf_low:
                self._retract_bound("high")

            # reset buffer
            self._buffer_high = []

    def _expand_bound(self, bound: str):
        """
        Push the chosen boundary outward by `adr_delta`.

        - In ADR mode the lower boundary is floored at `_MIN_PHYSICAL_MASS` (the mass
          must stay strictly positive); the upper boundary is unbounded.
        - In BADR mode both boundaries are hard-clamped against the user-provided
          `_mass_min_limit` / `_mass_max_limit` so the range can never exceed
          `mass_range`. An already-saturated side is skipped (the call is a no-op
          for that side). After the assignment, each boundary is re-checked against
          its limit with `np.isclose`; if a boundary now lies at its limit, the
          corresponding saturation flag (`_badr_lower_saturated` /
          `_badr_upper_saturated`) is set to True and the matching performance
          buffer is cleared.
        """
        if not self._badr_lower_saturated and bound == "low":
            if self.mode == "adr":
                # clamped against the physical-plausibility floor: mass must stay strictly positive
                self._mass_min = max(self._mass_min - self.adr_delta, self._MIN_PHYSICAL_MASS)
            else:   # badr
                # clamped against the user-provided lower limit
                self._mass_min = max(self._mass_min - self.adr_delta, self._mass_min_limit)

        elif not self._badr_upper_saturated and bound == "high":
            if self.mode == "adr":
                # unlimited expansion of upper bound
                self._mass_max = self._mass_max + self.adr_delta
            else:   # badr
                # clamped against the user-provided upper limit
                self._mass_max = min(self._mass_max + self.adr_delta, self._mass_max_limit)

        if self.mode == "badr":
            if np.isclose(self._mass_min, self._mass_min_limit):
                self._buffer_low = []
                self._badr_lower_saturated = True
            if np.isclose(self._mass_max, self._mass_max_limit):
                self._buffer_high = []
                self._badr_upper_saturated = True

    def _retract_bound(self, bound: str):
        """
        Pull the chosen boundary inward by `adr_delta`.

        Retraction is clamped against the opposite dynamic bound so that
        `_mass_min <= _mass_max` always holds (mode-agnostic). In BADR mode a side
        that has already saturated against its hard limit is treated as locked and
        will not be retracted — the corresponding flag (`_badr_lower_saturated` /
        `_badr_upper_saturated`) short-circuits the operation. In ADR mode the
        flags are never set, so retraction always proceeds.
        """
        if not self._badr_lower_saturated and bound == "low":
            # clamped against self._mass_max to preserve interval sanity
            self._mass_min = min(self._mass_min + self.adr_delta, self._mass_max)

        # bound = high
        elif not self._badr_upper_saturated and bound == "high":
            #  clamped against self._mass_min to preserve interval sanity
            self._mass_max = max(self._mass_max - self.adr_delta, self._mass_min)

    # -----------------------
    # Step
    # -----------------------
    def step(self, action):
        """
        Execute an environment step and track ADR / BADR performance.

        In ADR or BADR mode, this method accumulates rewards to calculate the total
        episode return. If the episode ends and a boundary was being tested for the
        current episode (`self._current_boundary is not None`), the result is stored
        in the appropriate side's buffer and `_update_boundaries` is invoked. In BADR
        mode, after a side saturates `_sample_mass_adr` stops choosing that side
        as a boundary.
        """
        # episode step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # update total episode return if adr
        if self.mode in ("adr", "badr"):
            self._episode_return += float(reward)

        # episode ended ?
        done = terminated or truncated

        # if episode ended and adr mode has done boundary sampling, update relative buffer
        if done and self.mode in ("adr", "badr") and self._current_boundary is not None:
            if self._current_boundary == "low":
                self._buffer_low.append(self._episode_return)
            else:  # "high"
                self._buffer_high.append(self._episode_return)
            self._update_boundaries()

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------
    def reset(self, **kwargs):
        """
        Reset the environment and apply a new randomized mass.

        Resets the episode return accumulator to zero, samples a new mass value via
        `_sample_mass` according to the active mode. In 'none' mode no mass
        change is applied.
        """
        # reset episode reward before sampling
        self._episode_return = 0.0

        # sample new mass, returns None if mode is "none"
        new_mass = self._sample_mass()

        # change object mass if UDR, ADR, or BADR mode (any mode that returns a mass)
        if new_mass is not None:
            # access and change object property
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]
            sim.physics_client.changeDynamics(bodyUniqueId=object_body_id,
                                              linkIndex=-1,
                                              mass=float(new_mass))

        # apply parent class reset
        return super().reset(**kwargs)