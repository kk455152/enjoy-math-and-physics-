import math
import unittest

from stepwise_em_model import DynamicState, ElectromagneticMemory, MagnetizationPoint, StepwiseElectromagneticModel


class SyntheticMap:
    def query_lab(self, material, N, R_mm, g_mm, D_mm, x_mm, y_mm, rotor_angle_rad):
        angle = math.atan2(y_mm, x_mm)
        return {"Fx_N": 0., "Fy_N": 0., "Fz_N": 0., "u_mm": 0., "v_mm": 0.,
                "magnet_index": 0, "force_policy": "synthetic_test"}

    def centerline_lab(self, material, N, R_mm, g_mm, D_mm, x_mm, y_mm,
                       rotor_angle_rad, eta, allow_unvalidated_z_interpolation=False):
        return {"Bx_T": .01*x_mm, "By_T": .01*y_mm, "Bz_T": .1,
                "Hx_Apm": 500*x_mm, "Hy_Apm": 300*y_mm, "Hz_Apm": 1000*eta}


class StepwiseTests(unittest.TestCase):
    def model(self, material="H62"):
        return StepwiseElectromagneticModel(SyntheticMap(), material=material, N=4,
                                            R_mm=41, g_mm=3, D_mm=3)

    def state(self, t, x=41.0, vx=.1):
        return DynamicState(t_s=t, x_mm=x, y_mm=0., vx_m_s=vx, vy_m_s=0.,
                            rotor_angle_rad=0., rotor_speed_rad_s=2.)

    def test_trial_is_transactional_and_h62_can_have_eddy_drag(self):
        m = self.model()
        s0 = self.state(0., x=41., vx=0.)
        memory = m.initialize(s0)
        original = memory.points[0].H_Apm[:]
        out, trial = m.trial_step(self.state(.001, x=41.1, vx=.1), memory)
        self.assertEqual(memory.t_s, 0.)
        self.assertEqual(memory.points[0].H_Apm, original)
        self.assertLessEqual(sum(a*b for a, b in zip(out["Feddy_low_order_xyz_N"], out["relative_velocity_xy_m_s"])), 1e-24)
        self.assertGreaterEqual(out["Jrms_joule_equivalent_Apm2"], 0.)
        self.assertEqual(out["J_vector_and_spatial_JxB"], "not_computed_by_this_reduced_order_stepper")
        self.assertEqual(trial.t_s, .001)

    def test_ferromagnetic_memory_advances_and_remains_bounded(self):
        m = self.model("GCr15_hardened")
        memory = m.initialize(self.state(0., x=41.))
        out, trial = m.trial_step(self.state(.001, x=41.5), memory)
        self.assertTrue(any(abs(v) > 0 for p in trial.points for v in p.M_irr_Apm))
        self.assertLess(max(abs(v) for p in trial.points for v in p.M_irr_Apm), m.ja.Ms_Apm)
        self.assertGreaterEqual(out["Physteresis_pin_est_W"], 0.)

    def test_time_must_advance(self):
        m = self.model()
        memory = m.initialize(self.state(1.))
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            m.trial_step(self.state(1.), memory)

    def test_continuous_gap_and_diameter_can_reach_the_field_interface(self):
        m = StepwiseElectromagneticModel(SyntheticMap(), material="H62", N=4,
                                        R_mm=41, g_mm=4.3, D_mm=5.2)
        memory=m.initialize(self.state(0.))
        out,trial=m.trial_step(self.state(.001,x=41.1),memory)
        self.assertTrue(all(math.isfinite(x) for x in out["Fmagnetic_xyz_N"]))
        self.assertAlmostEqual(m.a_m,.0026)
        self.assertEqual(trial.t_s,.001)

    def test_parameters_outside_shared_model_are_rejected(self):
        for gap,diameter in ((2.9,5.2),(6.1,5.2),(4.3,10.1),(float("nan"),5.2)):
            with self.assertRaises(ValueError):
                StepwiseElectromagneticModel(SyntheticMap(),material="H62",N=4,
                                             R_mm=41,g_mm=gap,D_mm=diameter)


if __name__ == "__main__":
    unittest.main()
