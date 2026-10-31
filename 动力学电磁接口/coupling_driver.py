"""Transactional predictor-corrector coupling for discrete dynamics steps.

The driver is deliberately agnostic about the mechanical integrator. The
caller supplies a pure one-step dynamics function; electromagnetic memory is
committed only by the caller after this function returns successfully.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from stepwise_em_model import DynamicState

MemoryT = TypeVar("MemoryT")
Vector3 = tuple[float, float, float]


class ElectromagneticStepper(Protocol[MemoryT]):
    def trial_step(self, state: DynamicState, memory: MemoryT) -> tuple[dict, MemoryT]: ...


class DynamicsTrial(Protocol):
    def __call__(self, state: DynamicState, magnetic_force_xyz_N: Vector3,
                 dt_s: float) -> DynamicState: ...


@dataclass(frozen=True)
class CoupledStep(Generic[MemoryT]):
    state: DynamicState
    force_record: dict
    candidate_memory: MemoryT
    iterations: int
    residual_norm_N: float


class CouplingConvergenceError(RuntimeError):
    def __init__(self, iterations: int, residual_norm_N: float):
        self.iterations = iterations
        self.residual_norm_N = residual_norm_N
        super().__init__(
            f"Predictor-corrector did not converge in {iterations} iterations "
            f"(force residual {residual_norm_N:.6g} N)"
        )


def _force_vector(record: dict) -> Vector3:
    value = record.get("Fmagnetic_xyz_N")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Electromagnetic record must contain Fmagnetic_xyz_N with 3 components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError("Electromagnetic force must be finite")
    return result  # type: ignore[return-value]


def predictor_corrector_step(
    electromagnetic: ElectromagneticStepper[MemoryT],
    state: DynamicState,
    committed_memory: MemoryT,
    dynamics_trial: DynamicsTrial,
    dt_s: float,
    *,
    previous_force_xyz_N: Vector3 = (0.0, 0.0, 0.0),
    force_abs_tol_N: float = 1e-8,
    force_rel_tol: float = 1e-3,
    relaxation: float = 0.5,
    max_iterations: int = 12,
) -> CoupledStep[MemoryT]:
    """Solve one partitioned time step without mutating the supplied memory.

    Every corrector iteration starts from the same mechanical state and
    committed electromagnetic memory. If convergence fails, the caller can
    reduce ``dt_s`` and retry using those unchanged inputs.
    """
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(force_abs_tol_N) or force_abs_tol_N < 0:
        raise ValueError("force_abs_tol_N must be finite and nonnegative")
    if not math.isfinite(force_rel_tol) or force_rel_tol < 0:
        raise ValueError("force_rel_tol must be finite and nonnegative")
    if not math.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError("relaxation must be in (0, 1]")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not all(math.isfinite(float(component)) for component in previous_force_xyz_N):
        raise ValueError("previous_force_xyz_N must be finite")

    guess = tuple(float(component) for component in previous_force_xyz_N)
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        candidate = dynamics_trial(state, guess, dt_s)
        if not isinstance(candidate, DynamicState):
            raise TypeError("dynamics_trial must return a DynamicState")
        target_time = state.t_s + dt_s
        time_tol = max(1e-12, 1e-9 * abs(target_time))
        if not math.isfinite(candidate.t_s) or abs(candidate.t_s - target_time) > time_tol:
            raise ValueError("dynamics_trial must return the state at state.t_s + dt_s")

        force_record, trial_memory = electromagnetic.trial_step(candidate, committed_memory)
        force = _force_vector(force_record)
        residual = math.sqrt(sum((force[i] - guess[i]) ** 2 for i in range(3)))
        tolerance = force_abs_tol_N + force_rel_tol * math.sqrt(sum(component**2 for component in force))
        if residual <= tolerance:
            return CoupledStep(candidate, force_record, trial_memory, iteration, residual)
        guess = tuple(guess[i] + relaxation * (force[i] - guess[i]) for i in range(3))

    raise CouplingConvergenceError(max_iterations, residual)
