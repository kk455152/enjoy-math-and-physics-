import csv
import math
import unittest
from pathlib import Path

from em_force_model import DEFAULT_DATA, ElectromagneticForceModel, ModelDomainError, TrajectoryState


ROOT = Path(__file__).resolve().parent


def make_model(**overrides):
    values = {
        "force_map": DEFAULT_DATA / "GCr15_hardened_static_force_dense.csv",
        "field_map": DEFAULT_DATA / "GCr15_hardened_centerline_BH.csv",
        "material": "GCr15_hardened",
        "N": 10,
        "R_mm": 80.0,
        "rho_mm": 85.0,
        "g_mm": 6.0,
        "D_mm": 6.0,
    }
    values.update(overrides)
    return ElectromagneticForceModel(**values)


def center_state(**overrides):
    values = {
        "t_s": 0.0,
        "x_mm": 0.0,
        "y_mm": 85.0,
        "vx_m_s": 0.0,
        "vy_m_s": 0.0,
        "Omega_rot_rad_s": 0.0,
    }
    values.update(overrides)
    return TrajectoryState(**values)


class ForceModelTests(unittest.TestCase):
    def test_static_force_matches_dense_map_at_sample(self):
        with (DEFAULT_DATA / "GCr15_hardened_static_force_dense.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        expected = next(row for row in rows if abs(float(row["vRel_mm"])) < 1e-8)
        result = make_model().evaluate(center_state(), 0.0)
        self.assertAlmostEqual(result["Fr_static_N"], float(expected["Fr_orbit_N"]), delta=1e-10)
        self.assertAlmostEqual(result["Ftheta_static_N"], float(expected["Ftheta_orbit_N"]), delta=1e-10)
        self.assertAlmostEqual(result["Fz_static_N"], float(expected["Fz_N"]), delta=1e-10)

    def test_rejects_outside_projection_and_other_geometry(self):
        with self.assertRaises(ModelDomainError):
            make_model().evaluate(center_state(x_mm=-14.8, y_mm=83.7), 0.0)
        with self.assertRaises(ModelDomainError):
            make_model().evaluate(center_state(y_mm=86.0), 0.0)

    def test_every_dense_position_sample_is_callable(self):
        with (DEFAULT_DATA / "GCr15_hardened_static_force_dense.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        model = make_model()
        for row in rows:
            v_mm = float(row["vRel_mm"])
            phase = math.asin(v_mm / 85.0)
            theta = math.pi / 2.0 + phase
            state = center_state(
                x_mm=85.0 * math.cos(theta),
                y_mm=85.0 * math.sin(theta),
            )
            result = model.evaluate(state, 0.0)
            self.assertAlmostEqual(result["vRel_mm"], v_mm, delta=1e-8)

    def test_eddy_force_dissipates_relative_translation(self):
        state = center_state(Omega_rot_rad_s=10.0)
        result = make_model().evaluate(state, 0.0)
        etheta = (-1.0, 0.0)
        vrel = (0.85, 0.0)
        feddy = (result["Feddy_x_N"], result["Feddy_y_N"])
        self.assertLessEqual(feddy[0] * vrel[0] + feddy[1] * vrel[1], 0.0)
        self.assertLessEqual(result["Feddy_tangent_N"] * (-0.85), 0.0)
        self.assertAlmostEqual(math.hypot(*etheta), 1.0)

    def test_hysteresis_surrogate_dissipates_generalized_power(self):
        state = center_state(Omega_rot_rad_s=10.0)
        result = make_model().evaluate(state, 0.0)
        vx_rel, vy_rel = 0.85, 0.0
        power = result["Fhyst_x_N"] * vx_rel + result["Fhyst_y_N"] * vy_rel
        power += result["Thyst_z_Nm"] * result["relative_spin_z_rad_s"]
        self.assertAlmostEqual(power, -result["Physt_est_W"], delta=max(1e-15, result["Physt_est_W"] * 1e-10))
        self.assertGreaterEqual(result["Physt_low_W"], 0.0)
        self.assertGreaterEqual(result["Physt_high_W"], result["Physt_low_W"])

    def test_hysteresis_uses_the_material_backbone_curve(self):
        model = make_model()
        self.assertAlmostEqual(model._backbone_B(1000.0), 2.0 * 1000.0 / (1000.0 + 53000.0), places=14)

    def test_optional_force_lag_uses_trajectory_history(self):
        states = [
            center_state(t_s=0.0, Omega_rot_rad_s=10.0),
            center_state(t_s=0.0005, Omega_rot_rad_s=10.0),
        ]
        lagged = make_model(tau_mag_s=0.01).evaluate_trajectory(states)
        instantaneous = make_model(tau_mag_s=0.0).evaluate_trajectory(states)
        self.assertNotAlmostEqual(lagged[1]["Fhyst_x_N"], instantaneous[1]["Fhyst_x_N"], places=8)

    def test_co_motion_has_no_reduced_dynamic_drag(self):
        state = center_state(vx_m_s=-0.85, Omega_rot_rad_s=10.0, omega_ball_z_rad_s=10.0)
        result = make_model().evaluate(state, 0.0)
        self.assertAlmostEqual(result["Feddy_tangent_N"], 0.0, places=15)
        self.assertAlmostEqual(result["Physt_est_W"], 0.0, places=15)

    def test_trajectory_integrates_variable_rotor_speed(self):
        model = make_model()
        states = [
            center_state(t_s=0.0, Omega_rot_rad_s=10.0),
            center_state(t_s=0.001, Omega_rot_rad_s=20.0),
        ]
        result = model.evaluate_trajectory(states)
        self.assertAlmostEqual(result[1]["rotor_angle_rad"], 0.015, places=12)


if __name__ == "__main__":
    unittest.main()
