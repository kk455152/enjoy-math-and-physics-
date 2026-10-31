"""Behavioral tests of continuous parameters, validation isolation and frame changes."""
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

from shared_parametric_map import (BOUNDS, FIELD_KEYS, POLICY, RBFGroup, SharedStaticForceMap,
                                   group_id, rotate, validate)


def analytical_point(c, curved=False, N=4):
    radial, phase, g, d = c
    # A labeled synthetic constitutive response with the leading volume scaling.
    curvature = 3*(g-3)*(g-6) if curved else 0.
    y = 1+.01*radial+.1*g+.03*d+curvature
    angle = math.pi/2 + phase*math.pi/N
    f = rotate([(d/6)**3*x for x in (.1*y, .2*y, -y)], angle)
    cosine, sine = math.cos(phase*math.pi/2), math.sin(phase*math.pi/2)
    b = rotate([(1+curvature)*x for x in (cosine,.2*sine,.5*cosine)], angle)
    h = rotate([(1+curvature)*x for x in (100*cosine,20*sine,50*cosine)], angle)
    fields = (b+h)*11
    return {"coordinates":list(c), "values":f+fields}


class SharedParameterTests(unittest.TestCase):
    def setUp(self):
        self.points = [analytical_point(c) for c in itertools.product(BOUNDS[0],(-1.,0.,1.),BOUNDS[2],BOUNDS[3])]

    def test_non_tabulated_gap_and_diameter_are_continuous(self):
        model = RBFGroup(self.points, "PC40")
        q = (2.1, -.8, 4.35, 5.25)
        for actual, expected in zip(model.predict(q), analytical_point(q)["values"]):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_independent_affine_response_passes(self):
        checks = [analytical_point((0., 0., 4.5, 6.5))]
        result = validate(RBFGroup(self.points, "PC40"), self.points, checks, 1)
        self.assertTrue(result["force_ready"])
        self.assertTrue(result["centerline_ready"])

    def test_missing_holdouts_never_pass(self):
        result = validate(RBFGroup(self.points, "PC40"), self.points, [], 16)
        self.assertFalse(result["complete"])

    def test_parameter_curvature_must_be_detected(self):
        training = [analytical_point(p["coordinates"], True) for p in self.points]
        checks = [analytical_point((0., 0., 4.5, 6.5), True)]
        result = validate(RBFGroup(training, "PC40"), training, checks, 1)
        self.assertFalse(result["force_ready"])
        self.assertFalse(result["centerline_ready"])

    def test_validation_points_cannot_leak_into_training(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate(RBFGroup(self.points, "PC40"), self.points, self.points[:1], 1)

    def test_missing_parameter_corner_rejects_coverage(self):
        with self.assertRaisesRegex(ValueError, "corners"):
            RBFGroup(self.points[:-1], "PC40")

    def test_extrapolation_rejected(self):
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            RBFGroup(self.points, "PC40").predict((0,0,6.01,6))

    def fixture(self, root, ready=True, N=4, R_mm=41):
        gid = group_id("PC40",N,R_mm)
        entry = {"group_id":gid, "file":gid+".json", "force_ready":ready,
                 "centerline_ready":ready, "validation":{}}
        (root/"manifest.json").write_text(json.dumps({"policy":POLICY, "template_sha256":"fixture", "groups":[entry]}))
        (root/entry["file"]).write_text(json.dumps({**entry, "template_sha256":"fixture",
                                                    "policy":POLICY, "training":[analytical_point(p["coordinates"],N=N) for p in self.points]}))

    def test_lab_rotation_and_alternating_field_polarity(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.fixture(root)
            model=SharedStaticForceMap(root)
            a=model.query_lab("PC40",4,41,4.3,5.2,0,41,0)
            b=model.query_lab("PC40",4,41,4.3,5.2,-41,0,0)
            self.assertAlmostEqual(b["Fx_N"],-a["Fy_N"])
            self.assertAlmostEqual(b["Fy_N"],a["Fx_N"])
            fa=model.centerline_lab("PC40",4,41,4.3,5.2,0,41,0,0)
            fb=model.centerline_lab("PC40",4,41,4.3,5.2,-41,0,0,0)
            self.assertAlmostEqual(fb["By_T"],-fa["Bx_T"])
            self.assertAlmostEqual(fb["Bz_T"],-fa["Bz_T"])

    def test_pending_group_cannot_drive_dynamics(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.fixture(root,False)
            with self.assertRaisesRegex(RuntimeError,"not validated"):
                SharedStaticForceMap(root).query_lab("PC40",4,41,4.3,5.2,0,41,0)

    def test_between_magnets_is_included_in_radial_band(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.fixture(root)
            q=SharedStaticForceMap(root).query_lab("PC40",4,41,4.3,5.2,
                                                   -41*.5,41*math.sqrt(3)/2,0)
            self.assertGreater(abs(q["v_mm"]),5)
            self.assertAlmostEqual(q["rho_mm"],41)

    def test_whole_orbit_force_and_field_remain_continuous_at_sector_seams(self):
        with tempfile.TemporaryDirectory() as t:
            for n,r in ((4,41),(10,96)):
                root=Path(t)/str(n)
                root.mkdir()
                self.fixture(root,N=n,R_mm=r)
                model=SharedStaticForceMap(root)
                for j in range(n):
                    sides=[]
                    for epsilon in (-1e-10,1e-10):
                        angle=math.pi/2+(j+.5)*2*math.pi/n+epsilon
                        args=("PC40",n,r,4.3,5.2,r*math.cos(angle),r*math.sin(angle),0)
                        f=model.query_lab(*args)
                        b=model.centerline_lab(*args,0)
                        sides.append([f[k] for k in ("Fx_N","Fy_N","Fz_N")]+[b[k] for k in FIELD_KEYS])
                    for a,b in zip(*sides):
                        self.assertAlmostEqual(a,b,places=6)

    def test_continuous_map_can_advance_existing_dynamics_memory_at_a_seam(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"动力学电磁接口"))
        from stepwise_em_model import DynamicState, StepwiseElectromagneticModel
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.fixture(root)
            model=StepwiseElectromagneticModel(SharedStaticForceMap(root),material="PC40",
                                              N=4,R_mm=41,g_mm=4.3,D_mm=5.2)
            def state(time,angle):
                return DynamicState(t_s=time,x_mm=41*math.cos(angle),y_mm=41*math.sin(angle),
                                    vx_m_s=.001,vy_m_s=.002,rotor_angle_rad=0,rotor_speed_rad_s=2)
            memory=model.initialize(state(0,3*math.pi/4-1e-4))
            force,candidate=model.trial_step(state(.001,3*math.pi/4+1e-4),memory)
            self.assertTrue(all(math.isfinite(v) for v in force["Fmagnetic_xyz_N"]))
            self.assertEqual(memory.t_s,0)
            self.assertEqual(candidate.t_s,.001)

    def test_radial_band_cannot_be_replaced_by_rectangular_bounds(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError,"rho_minus_R_mm"):
                SharedStaticForceMap(root).query_local("PC40",4,41,4.3,5.2,15,5)

    def test_old_rectangular_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/"manifest.json").write_text(json.dumps({"policy":"old_rectangular"}))
            with self.assertRaisesRegex(RuntimeError,"rectangular"):
                SharedStaticForceMap(root)

    def test_layout_and_material_are_not_interpolated(self):
        with self.assertRaises(ValueError):
            group_id("PC40",6,70)
        with self.assertRaises(ValueError):
            group_id("imaginary_material",4,41)


if __name__ == "__main__":
    unittest.main()
