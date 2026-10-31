"""Transactional force AND torque coupling with an external dynamics integrator."""
from dataclasses import dataclass
import math
from typing import Generic, TypeVar
from stepwise_em_model import DynamicState

MemoryT=TypeVar("MemoryT")


@dataclass(frozen=True)
class FullCoupledStep(Generic[MemoryT]):
    state: DynamicState
    force_record: dict
    candidate_memory: MemoryT
    iterations: int
    force_residual_N: float
    torque_residual_Nm: float


class FullCouplingConvergenceError(RuntimeError):
    pass


def _vector(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3:
        raise ValueError("A finite 3-vector is required")
    result=tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("A finite 3-vector is required")
    return result


def predictor_corrector_full_step(electromagnetic,state,committed_memory,dynamics_trial,
        dt_s,*,previous_force_xyz_N=(0.,0.,0.),previous_torque_xyz_Nm=(0.,0.,0.),
        force_abs_tol_N=1e-7,force_rel_tol=1e-3,
        torque_abs_tol_Nm=1e-9,torque_rel_tol=1e-3,
        relaxation=.5,max_iterations=12):
    """dynamics_trial(state,Fxyz,Txyz,dt) returns an uncommitted DynamicState.

    Every corrector reuses the SAME committed EM memory. Both total force and
    total torque must converge; no hidden update or acceptance occurs.
    """
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dt_s must be positive and finite")
    if not math.isfinite(relaxation) or not 0<relaxation<=1:
        raise ValueError("relaxation must lie in (0,1]")
    if not isinstance(max_iterations,int) or max_iterations<1:
        raise ValueError("max_iterations must be a positive integer")
    for tol in (force_abs_tol_N,force_rel_tol,torque_abs_tol_Nm,torque_rel_tol):
        if not math.isfinite(tol) or tol<0:
            raise ValueError("Coupling tolerances must be finite and nonnegative")
    force=_vector(previous_force_xyz_N);torque=_vector(previous_torque_xyz_Nm)
    for iteration in range(1,max_iterations+1):
        end=dynamics_trial(state,force,torque,dt_s)
        if not isinstance(end,DynamicState) or not math.isclose(end.t_s,state.t_s+dt_s,rel_tol=1e-9,abs_tol=1e-12):
            raise ValueError("Dynamics must return a DynamicState at the target time")
        record,candidate=electromagnetic.trial_step(end,committed_memory)
        fnew=_vector(record.get("Fmagnetic_xyz_N"));tnew=_vector(record.get("Tmagnetic_xyz_Nm"))
        rf=math.dist(force,fnew);rt=math.dist(torque,tnew)
        if (rf<=force_abs_tol_N+force_rel_tol*math.sqrt(sum(v*v for v in fnew)) and
                rt<=torque_abs_tol_Nm+torque_rel_tol*math.sqrt(sum(v*v for v in tnew))):
            return FullCoupledStep(end,record,candidate,iteration,rf,rt)
        force=tuple(a+relaxation*(b-a) for a,b in zip(force,fnew))
        torque=tuple(a+relaxation*(b-a) for a,b in zip(torque,tnew))
    raise FullCouplingConvergenceError(f"No coupled convergence after {max_iterations} trials; "
            f"force residual={rf:g} N, torque residual={rt:g} N m. Keep committed memory and reduce dt.")
