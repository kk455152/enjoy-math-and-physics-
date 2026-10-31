"""Isolated synthetic I/O tests; these do not certify COMSOL physics accuracy."""
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from full_transient_stepper import ComsolTransientStepper, _source_digest, _spin, _hermite
from stepwise_em_model import DynamicState
from comsol_backend import validate


SETTINGS={"material":"GCr15_hardened","N":4,"R_mm":41.,"rho_mm":41.,
          "g_mm":3.,"D_mm":6.,"delta_deg":0.,"Omega_rad_s":10.,
          "dt_s":.001,"Rair_mm":180.,"mesh_scale":3.,"rtol":.002,"threads":1}


class SyntheticRunner:
    """Writes explicitly synthetic endpoint tables only inside temporary tests."""
    def __init__(self):self.calls=[]
    def __call__(self,settings,motion,output,**kwargs):
        self.calls.append((copy.deepcopy(settings),copy.deepcopy(motion),kwargs))
        output.mkdir()
        t=0 if kwargs["initialize_only"] else motion[-1][0]
        values={"t_s":t,"min_relative_mesh_volume":1.,"Jrms_A_m2":123.,"Pjoule_W":.04}
        for a in "xyz":
            values[f"F{a}_total_N"]=.2+t
            values[f"T{a}_total_Nm"]=.03
            values[f"JxB_integral_{a}_N"]=.001
            values[f"Mmean_{a}_A_m"]=100.
        (output/"total_force_history.txt").write_text("% "+" ".join(values)+"\n"+" ".join(map(str,values.values()))+"\n",encoding="utf-8")
        (output/"run_status.json").write_text(json.dumps({"status":"stationary_solved" if kwargs["initialize_only"] else "solved","template_sha256":_source_digest(),"synthetic_test_data":True}),encoding="utf-8")


class FullTransientTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.runner=SyntheticRunner()
        self.model=ComsolTransientStepper(SETTINGS,self.tmp.name,runner=self.runner)
        self.initial=DynamicState(0,0,41,0,0,0,0)

    def tearDown(self):self.tmp.cleanup()

    def test_trial_does_not_advance_committed_memory(self):
        record,old=self.model.initialize(self.initial)
        snapshot=asdict(old)
        end=replace(self.initial,t_s=.002,x_mm=.2,vx_m_s=.1,rotor_angle_rad=.03,rotor_speed_rad_s=10)
        result,trial=self.model.trial_step(end,old)
        self.assertEqual(asdict(old),snapshot)
        self.assertEqual(len(trial.states),2)
        self.assertEqual(result["Fmagnetic_xyz_N"],(.202,)*3)
        self.assertEqual(result["backend"],"COMSOL_spatial_Maxwell_vector_JA_full_causal_replay")
        self.assertFalse(result["numerically_validated"])
        self.assertEqual(trial.motion[-1][1:3],(.2,0))
        self.assertEqual(trial.motion[-1][3],.03)

    def test_corrector_replays_same_base_and_cache_reuses_identical_trial(self):
        _,base=self.model.initialize(self.initial)
        end=replace(self.initial,t_s=.001,x_mm=.1)
        _,a=self.model.trial_step(end,base)
        _,b=self.model.trial_step(end,base)
        self.assertEqual(a,b)
        self.assertEqual(len(self.runner.calls),2)
        _,c=self.model.trial_step(replace(end,x_mm=.12),base)
        self.assertEqual(len(c.states),2)
        self.assertEqual(base.states,(self.initial,))

    def test_accepted_history_prefix_is_preserved(self):
        _,base=self.model.initialize(self.initial)
        _,first=self.model.trial_step(replace(self.initial,t_s=.001,x_mm=.1),base)
        _,second=self.model.trial_step(replace(self.initial,t_s=.002,x_mm=.2),first)
        self.assertEqual(second.motion[:len(first.motion)],first.motion)
        self.assertEqual(len(second.states),3)

    def test_restore_and_tamper_detection(self):
        _,base=self.model.initialize(self.initial)
        restored=self.model.load_memory(Path(base.run_directory)/"coupling_memory.json")
        self.assertEqual(base,restored)
        tampered=replace(base,motion=((0,1,0,0,1,0,0,0),))
        with self.assertRaisesRegex(ValueError,"digest"):
            self.model.trial_step(replace(self.initial,t_s=.001),tampered)
        with self.assertRaisesRegex(ValueError,"another"):
            self.model.trial_step(replace(self.initial,t_s=.001),replace(base,source_sha256="wrong"))

    def test_spin_is_normalized_and_uses_lab_frame(self):
        q=_spin((1.,0.,0.,0.),(math.pi,0,0),1)
        self.assertAlmostEqual(q[0],0)
        self.assertAlmostEqual(q[1],1)
        self.assertAlmostEqual(sum(x*x for x in q),1)
        q2=_spin(q,(0,math.pi,0),1)
        self.assertAlmostEqual(q2[3],-1)

    def test_hermite_preserves_endpoints_and_velocity(self):
        self.assertEqual(_hermite(2,5,7,9,0,.2),2)
        self.assertEqual(_hermite(2,5,7,9,1,.2),5)
        h=1e-6
        self.assertAlmostEqual((_hermite(2,5,7,9,h,.2)-2)/(h*.2),7,places=3)

    def test_domain_checks_all_orbit_angles_and_segment_interior(self):
        # Tangential gap positions are permitted: radius, not rectangular projection.
        angle=.3
        validate(SETTINGS,[(0,0,0),(1,-41*math.sin(angle),41*math.cos(angle)-41)])
        with self.assertRaisesRegex(ValueError,"radial"):
            validate(SETTINGS,[(0,0,0),(1,0,30)])
        with self.assertRaisesRegex(ValueError,"inner"):
            validate(SETTINGS,[(0,0,0),(1,0,-82)])

    def test_quaternion_sign_and_nonfinite_states(self):
        with self.assertRaisesRegex(ValueError,"signs"):
            validate(SETTINGS,[(0,0,0,0,1,0,0,0),(1,0,0,0,-1,0,0,0)])
        with self.assertRaises(ValueError):
            self.model.initialize(replace(self.initial,vx_m_s=math.nan))
        _,base=self.model.initialize(self.initial)
        with self.assertRaises(ValueError):self.model.trial_step(self.initial,base)

    def test_reversible_branch_retains_current_history_and_cannot_reuse_JA_memory(self):
        _,base=self.model.initialize(self.initial)
        reversible=ComsolTransientStepper({**SETTINGS,"magnetization_model":"reversible_bh",
                                          "bh_curve_source":"ja_anhysteretic"},
                                         Path(self.tmp.name)/"reversible",runner=self.runner)
        row,rb=reversible.initialize(self.initial)
        self.assertFalse(row["hysteresis_retained"])
        self.assertTrue(row["transient_eddy_currents_retained"])
        self.assertEqual(row["magnetization_model"],"reversible_bh")
        self.assertEqual(self.runner.calls[-1][0]["bh_curve_source"],"ja_anhysteretic")
        end=replace(self.initial,t_s=.001,x_mm=.1)
        with self.assertRaisesRegex(ValueError,"another"):
            reversible.trial_step(end,base)
        _,trial=reversible.trial_step(end,rb)
        self.assertEqual(trial.motion[:len(rb.motion)],rb.motion)

    def test_invalid_constitutive_choices_are_rejected(self):
        motion=[(0,0,0),(1,0,0)]
        for extra in ({"magnetization_model":"linear"}, {"bh_curve_source":"invented"},
                      {"magnetization_model":"reversible_bh","initialization_mode":"uniform_M"},
                      {"magnetization_model":"reversible_bh","M0x_A_m":1}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                validate({**SETTINGS,**extra},motion)


if __name__=="__main__":unittest.main()
