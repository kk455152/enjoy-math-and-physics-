import unittest
from dataclasses import dataclass

from coupling_driver import CouplingConvergenceError, predictor_corrector_step
from stepwise_em_model import DynamicState


@dataclass
class Memory:
    t_s: float
    accepted_count: int = 0


class LinearForceModel:
    def trial_step(self, state, memory):
        record = {"Fmagnetic_xyz_N": [1e-5 * state.x_mm, 0.0, 0.0]}
        return record, Memory(state.t_s, memory.accepted_count + 1)


def initial_state():
    return DynamicState(0.0, 10.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def linear_dynamics(state, force, dt):
    vx = state.vx_m_s + force[0] * dt
    return DynamicState(state.t_s + dt, state.x_mm + vx * dt * 1000,
                        state.y_mm, vx, state.vy_m_s, state.rotor_angle_rad,
                        state.rotor_speed_rad_s)


class CouplingDriverTests(unittest.TestCase):
    def test_corrector_converges_and_returns_candidate_memory(self):
        state = initial_state()
        memory = Memory(0.0)
        step = predictor_corrector_step(
            LinearForceModel(), state, memory, linear_dynamics, 0.01,
            max_iterations=20,
        )
        self.assertGreater(step.iterations, 1)
        self.assertEqual(step.state.t_s, 0.01)
        self.assertEqual(step.candidate_memory.t_s, 0.01)
        self.assertEqual(memory.t_s, 0.0)
        self.assertEqual(memory.accepted_count, 0)

    def test_nonconvergence_leaves_committed_memory_untouched(self):
        state = initial_state()
        memory = Memory(0.0)
        with self.assertRaises(CouplingConvergenceError):
            predictor_corrector_step(
                LinearForceModel(), state, memory, linear_dynamics, 0.01,
                max_iterations=1, force_abs_tol_N=0.0, force_rel_tol=0.0,
            )
        self.assertEqual(memory.t_s, 0.0)
        self.assertEqual(memory.accepted_count, 0)

    def test_dynamics_trial_must_return_exact_step_time(self):
        state = initial_state()
        with self.assertRaisesRegex(ValueError, "state.t_s \+ dt_s"):
            predictor_corrector_step(
                LinearForceModel(), state, Memory(0.0),
                lambda s, f, dt: s, 0.01,
            )


if __name__ == "__main__":
    unittest.main()
