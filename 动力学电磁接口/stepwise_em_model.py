"""Per-step magnetic-force callback for an externally integrated ball model.

This is an explicitly reduced-order model. It combines the new validated
static-map interface, a low-magnetic-Reynolds eddy-drag approximation, and a
Jiles-Atherton-like magnetization memory estimate. It does not solve volumetric
J or a moving-domain Maxwell problem; the output labels those limits.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Protocol

MU0 = 4e-7 * math.pi
ETAS = tuple(round(-1 + i * .2, 8) for i in range(11))
FIELD_KEYS = ("Bx_T", "By_T", "Bz_T", "Hx_Apm", "Hy_Apm", "Hz_Apm")


@dataclass(frozen=True)
class DynamicState:
    t_s: float
    x_mm: float
    y_mm: float
    vx_m_s: float
    vy_m_s: float
    rotor_angle_rad: float
    rotor_speed_rad_s: float
    omega_ball_z_rad_s: float = 0.0
    omega_ball_x_rad_s: float = 0.0
    omega_ball_y_rad_s: float = 0.0


@dataclass(frozen=True)
class JAParameters:
    Ms_Apm: float
    a_Apm: float
    k_Apm: float
    c: float
    alpha: float
    Hc_Apm: float
    Br_T: float


# Engineering starting estimates from the supplied backbone curves and
# hysteresis estimates. They are not fitted measured loops.
JA_PARAMETERS = {
    "PC40": JAParameters(.60 / MU0, 70., 14.3, .10, 1e-4, 14.3, .50),
    "GCr15_hardened": JAParameters(2.0 / MU0, 18_300., 2_900., .10, 1e-4, 2_900., .70),
    "GCr15_annealed": JAParameters(2.0 / MU0, 5_400., 400., .10, 1e-4, 400., 1.35),
}
CONDUCTIVITY = {"H62": 15.7e6, "PC40": 1.0,
                "GCr15_hardened": 5.3e6, "GCr15_annealed": 5.3e6}


@dataclass
class MagnetizationPoint:
    H_Apm: list[float] = field(default_factory=lambda: [0., 0., 0.])
    M_irr_Apm: list[float] = field(default_factory=lambda: [0., 0., 0.])
    M_Apm: list[float] = field(default_factory=lambda: [0., 0., 0.])


@dataclass
class ElectromagneticMemory:
    t_s: float | None = None
    points: list[MagnetizationPoint] = field(default_factory=list)
    initial_condition: str = "demagnetized_zero_M_approximation"


class FieldMap(Protocol):
    def query_lab(self, material, N, R_mm, g_mm, D_mm, x_mm, y_mm, rotor_angle_rad): ...
    def centerline_lab(self, material, N, R_mm, g_mm, D_mm, x_mm, y_mm,
                       rotor_angle_rad, eta, allow_unvalidated_z_interpolation=False): ...


def _langevin(x: float) -> float:
    if abs(x) < 1e-4:
        return x / 3 - x**3 / 45 + 2*x**5 / 945
    if x > 40:
        return 1 - 1/x
    if x < -40:
        return -1 - 1/x
    return 1 / math.tanh(x) - 1 / x


def _advance_component(H_new: float, old: MagnetizationPoint, axis: int,
                       p: JAParameters) -> tuple[float, float]:
    H_old, mirr = old.H_Apm[axis], old.M_irr_Apm[axis]
    delta_H = H_new - H_old
    if abs(delta_H) < 1e-14:
        he = H_new + p.alpha * old.M_Apm[axis]
        man = p.Ms_Apm * _langevin(he / p.a_Apm)
        return mirr, p.c * man + (1 - p.c) * mirr
    step_limit = max(1.0, min(p.a_Apm, p.k_Apm) / 8.0)
    nstep = min(2000, max(1, math.ceil(abs(delta_H) / step_limit)))
    h = H_old
    dh = delta_H / nstep
    direction = 1. if delta_H > 0 else -1.
    m = old.M_Apm[axis]
    for _ in range(nstep):
        he = h + p.alpha * m
        man = p.Ms_Apm * _langevin(he / p.a_Apm)
        diff = man - mirr
        denom = direction * p.k_Apm - p.alpha * diff
        floor = max(1e-9, .05 * p.k_Apm)
        if abs(denom) < floor:
            denom = math.copysign(floor, denom if denom else direction)
        dmirr_dh = diff / denom
        mirr = max(-p.Ms_Apm, min(p.Ms_Apm, mirr + dmirr_dh * dh))
        h += dh
        he = h + p.alpha * m
        man = p.Ms_Apm * _langevin(he / p.a_Apm)
        m = p.c * man + (1 - p.c) * mirr
    return mirr, m


class StepwiseElectromagneticModel:
    """Stateless trial evaluator; caller commits returned memory on accepted steps."""

    def __init__(self, field_map: FieldMap, *, material: str, N: int, R_mm: float,
                 g_mm: float, D_mm: float, plate_mm: float = 1.0,
                 gradient_step_mm: float = .2):
        if material not in CONDUCTIVITY:
            raise ValueError(f"Unsupported material: {material}")
        if N not in (4, 10):
            raise ValueError("Only the requested computed layouts N=4 and N=10 are supported")
        if (N == 4 and R_mm != 41) or (N == 10 and R_mm != 96):
            raise ValueError("The current campaign includes only N=4/R=41 and N=10/R=96 mm")
        if not math.isfinite(g_mm) or not math.isfinite(D_mm) or not (3 <= g_mm <= 6 and 3 <= D_mm <= 10):
            raise ValueError("The shared parameter model supports g in [3,6] mm and D in [3,10] mm")
        if plate_mm != 1.0:
            raise ValueError("The current maps assume a 1 mm nonmagnetic, nonconducting plate")
        self.field_map, self.material = field_map, material
        self.N, self.R_mm, self.g_mm, self.D_mm = N, R_mm, g_mm, D_mm
        self.sigma, self.a_m = CONDUCTIVITY[material], D_mm / 2000
        self.volume = 4 * math.pi * self.a_m**3 / 3
        self.gradient_step_mm = gradient_step_mm
        self.ja = JA_PARAMETERS.get(material)

    @staticmethod
    def _validate_state(s: DynamicState):
        values = (s.t_s, s.x_mm, s.y_mm, s.vx_m_s, s.vy_m_s, s.rotor_angle_rad,
                  s.rotor_speed_rad_s, s.omega_ball_z_rad_s, s.omega_ball_x_rad_s,
                  s.omega_ball_y_rad_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Dynamic state must contain only finite values")

    def _field_profile(self, s: DynamicState):
        rows = []
        for eta in ETAS:
            q = self.field_map.centerline_lab(
                self.material, self.N, self.R_mm, self.g_mm, self.D_mm,
                s.x_mm, s.y_mm, s.rotor_angle_rad, eta,
            )
            rows.append({key: float(q[key]) for key in FIELD_KEYS})
        return rows

    def initialize(self, s: DynamicState, *, initial_M_Apm=None,
                   initial_condition="demagnetized_zero_M_approximation") -> ElectromagneticMemory:
        self._validate_state(s)
        fields = self._field_profile(s)
        memory = ElectromagneticMemory(t_s=s.t_s, initial_condition=initial_condition)
        for i, row in enumerate(fields):
            H = [row[k] for k in ("Hx_Apm", "Hy_Apm", "Hz_Apm")]
            M = list(initial_M_Apm[i]) if initial_M_Apm is not None else [0., 0., 0.]
            if len(M) != 3:
                raise ValueError("initial_M_Apm must contain one 3-vector per eta sample")
            memory.points.append(MagnetizationPoint(H, list(M), list(M)))
        return memory

    def _relative_coordinates(self, s: DynamicState):
        x, y = s.x_mm / 1000, s.y_mm / 1000
        theta = math.atan2(y, x)
        er, et = (math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))
        omega_orbit = (x*s.vy_m_s - y*s.vx_m_s) / max(x*x + y*y, 1e-30)
        vrel = (s.vx_m_s + s.rotor_speed_rad_s*y,
                s.vy_m_s - s.rotor_speed_rad_s*x)
        omega_rel = s.omega_ball_z_rad_s - s.rotor_speed_rad_s
        return er, et, vrel, omega_orbit, omega_rel

    def _eddy_matrix(self, s: DynamicState):
        eps = self.gradient_step_mm
        origin = self._field_profile(s)
        gradients = []
        for axis, (dx, dy) in enumerate(((eps, 0.), (0., eps))):
            samples = []
            for sign in (-1., 1.):
                shifted = DynamicState(**{**s.__dict__, "x_mm": s.x_mm + sign*dx,
                                           "y_mm": s.y_mm + sign*dy})
                try:
                    samples.append(self._field_profile(shifted))
                except (ValueError, RuntimeError):
                    samples.append(None)
            delta_m = 2*eps/1000
            derivative = []
            for i, base in enumerate(origin):
                if samples[0] is not None and samples[1] is not None:
                    lo, hi = samples[0][i], samples[1][i]
                    derivative.append(tuple((hi[k]-lo[k])/delta_m for k in ("Bx_T", "By_T", "Bz_T")))
                elif samples[1] is not None:
                    derivative.append(tuple((samples[1][i][k]-base[k])/(eps/1000) for k in ("Bx_T", "By_T", "Bz_T")))
                elif samples[0] is not None:
                    derivative.append(tuple((base[k]-samples[0][i][k])/(eps/1000) for k in ("Bx_T", "By_T", "Bz_T")))
                else:
                    raise ValueError("Cannot compute local B gradient without leaving the validated map")
            gradients.append(derivative)

        # Sphere cross-section weighting: A_jk = <sum_i dB_i/dx_j dB_i/dx_k>_V.
        raw_weights = [1 - eta*eta for eta in ETAS]
        total = sum(raw_weights)
        weights = [w/total for w in raw_weights]
        a00 = sum(w*sum(component**2 for component in gx_eta)
                  for w, gx_eta in zip(weights, gradients[0]))
        a11 = sum(w*sum(component**2 for component in gy_eta)
                  for w, gy_eta in zip(weights, gradients[1]))
        a01 = sum(w*sum(gx*gy for gx, gy in zip(gx_eta, gy_eta))
                  for w, gx_eta, gy_eta in zip(weights, gradients[0], gradients[1]))
        return ((a00, a01), (a01, a11))

    def trial_step(self, s: DynamicState, memory: ElectromagneticMemory):
        self._validate_state(s)
        if memory.t_s is None or len(memory.points) != len(ETAS):
            raise ValueError("Initialize electromagnetic memory at the first dynamic state")
        dt = s.t_s - memory.t_s
        if dt <= 0:
            raise ValueError("Time must strictly increase from the committed electromagnetic state")
        force_static = self.field_map.query_lab(
            self.material, self.N, self.R_mm, self.g_mm, self.D_mm,
            s.x_mm, s.y_mm, s.rotor_angle_rad,
        )
        field_now = self._field_profile(s)
        next_memory = copy.deepcopy(memory)
        next_memory.t_s = s.t_s
        dmirr_by_eta = []
        if self.ja:
            for i, (point, row) in enumerate(zip(memory.points, field_now)):
                h = [row[k] for k in ("Hx_Apm", "Hy_Apm", "Hz_Apm")]
                new_point = MagnetizationPoint(H_Apm=h, M_irr_Apm=list(point.M_irr_Apm),
                                               M_Apm=list(point.M_Apm))
                dmirr = []
                for axis in range(3):
                    new_point.M_irr_Apm[axis], new_point.M_Apm[axis] = _advance_component(
                        h[axis], point, axis, self.ja)
                    dmirr.append(new_point.M_irr_Apm[axis] - point.M_irr_Apm[axis])
                dmirr_by_eta.append(dmirr)
                next_memory.points[i] = new_point

        er, et, vrel, omega_orbit, omega_rel = self._relative_coordinates(s)
        tensor = self._eddy_matrix(s)
        ce = (2*math.pi/15) * self.sigma * self.a_m**5
        feddy = (-ce*(tensor[0][0]*vrel[0] + tensor[0][1]*vrel[1]),
                 -ce*(tensor[1][0]*vrel[0] + tensor[1][1]*vrel[1]), 0.)
        peddy = max(0., -(feddy[0]*vrel[0] + feddy[1]*vrel[1]))
        j_equiv = math.sqrt(self.sigma*peddy/self.volume) if self.sigma > 0 and self.volume > 0 else 0.

        physt = 0.
        if self.ja:
            weights = [1-eta*eta for eta in ETAS]
            normalizer = sum(weights)
            irreversible_rate = sum(w*sum(abs(q) for q in dm)
                                    for w, dm in zip(weights, dmirr_by_eta)) / normalizer / dt
            # Pinning-work estimate. It is nonnegative and carries the
            # irreversible magnetization history, but is not a measured loop.
            physt = MU0*self.ja.k_Apm*irreversible_rate*self.volume
        denominator = vrel[0]**2 + vrel[1]**2 + self.a_m**2*omega_rel**2
        if denominator > 0:
            fh = (-physt*vrel[0]/denominator, -physt*vrel[1]/denominator, 0.)
            th = -physt*self.a_m**2*omega_rel/denominator
        else:
            fh, th = (0., 0., 0.), 0.

        ftotal = tuple(float(force_static[k]) + feddy[i] + fh[i]
                       for i, k in enumerate(("Fx_N", "Fy_N", "Fz_N")))
        result = {
            "t_s": s.t_s,
            "u_mm": force_static["u_mm"], "v_mm": force_static["v_mm"],
            "magnet_index": force_static["magnet_index"],
            "Fstatic_xyz_N": [force_static[k] for k in ("Fx_N", "Fy_N", "Fz_N")],
            "Feddy_low_order_xyz_N": list(feddy),
            "Peddy_low_order_W": peddy,
            "Jrms_joule_equivalent_Apm2": j_equiv,
            "Fhysteresis_loss_xyz_N": list(fh),
            "Thysteresis_loss_z_Nm": th,
            "Physteresis_pin_est_W": physt,
            "Fmagnetic_xyz_N": list(ftotal),
            "Teddy_about_ball_center_Nm": "not_resolved",
            "JA_M_irr_Apm_by_eta": [p.M_irr_Apm for p in next_memory.points] if self.ja else [],
            "JA_state_feedback_into_BH_field": "not_coupled; memory currently drives only the pinning-loss force estimate",
            "relative_velocity_xy_m_s": list(vrel),
            "relative_orbit_rate_rad_s": omega_orbit - s.rotor_speed_rad_s,
            "relative_spin_z_rad_s": omega_rel,
            "non_z_ball_spin": "not_in_eddy_model" if (s.omega_ball_x_rad_s != 0 or s.omega_ball_y_rad_s) else "not_requested",
            "static_map_force_policy": force_static["force_policy"],
            "J_vector_and_spatial_JxB": "not_computed_by_this_reduced_order_stepper",
            "model_status": "static_map_plus_low_Re_eddy_tensor_and_approximate_JA_history_loss",
            "accepted_step_action": "replace_memory_with_returned_memory_only_after_dynamics_accepts_step",
        }
        return result, next_memory


def semi_implicit_euler(state: DynamicState, magnetic_force_xyz_N, dt_s: float,
                        mass_kg: float, external_force_xyz_N=(0., 0., 0.)) -> DynamicState:
    """Minimal adapter example; production contact/friction remains caller-owned."""
    if not math.isfinite(mass_kg) or mass_kg <= 0 or dt_s <= 0:
        raise ValueError("mass and timestep must be positive")
    ax = (magnetic_force_xyz_N[0] + external_force_xyz_N[0]) / mass_kg
    ay = (magnetic_force_xyz_N[1] + external_force_xyz_N[1]) / mass_kg
    vx, vy = state.vx_m_s + ax*dt_s, state.vy_m_s + ay*dt_s
    return DynamicState(
        t_s=state.t_s + dt_s,
        x_mm=state.x_mm + vx*dt_s*1000,
        y_mm=state.y_mm + vy*dt_s*1000,
        vx_m_s=vx, vy_m_s=vy,
        rotor_angle_rad=state.rotor_angle_rad + state.rotor_speed_rad_s*dt_s,
        rotor_speed_rad_s=state.rotor_speed_rad_s,
        omega_ball_z_rad_s=state.omega_ball_z_rad_s,
        omega_ball_x_rad_s=state.omega_ball_x_rad_s,
        omega_ball_y_rad_s=state.omega_ball_y_rad_s,
    )
