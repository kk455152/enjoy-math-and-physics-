"""Isolated synthetic numerical/comparison tests; no native evidence is generated."""
import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import full_transient_stepper
from compare_transient_runs import _statistics, compare_runs
from compare_dynamics_speed import compare_speed


class HysteresisComparisonTests(unittest.TestCase):
    def test_vector_impulse_keeps_sign_and_components(self):
        stats,delta=_statistics([0.,1.],[(1.,2.,3.)]*2,[(2.,0.,3.)]*2)
        self.assertEqual(stats["difference_integral_xyz"],[1.,-2.,0.])
        self.assertAlmostEqual(stats["difference_rms"],math.sqrt(5))
        self.assertEqual(delta,[(1.,-2.,0.)]*2)

    def test_comparison_refuses_confounded_or_changed_sources(self):
        # _load is mocked explicitly; these values never become native run files.
        status={"template_sha256":"synthetic_source"}
        params={"material":"GCr15_hardened","magnetization_model":"jiles_atherton","g_mm":3}
        first=(Path("synthetic_A"),status,params,[[0,0,0],[1,0,0]],[])
        second=(Path("synthetic_B"),status,{**params,"magnetization_model":"reversible_bh","g_mm":4},first[3],[])
        with patch("compare_transient_runs._load",side_effect=[first,second]):
            with self.assertRaisesRegex(ValueError,"parameters differ"):
                compare_runs("synthetic_A","synthetic_B")
        changed=list(copy.deepcopy(second));changed[1]={"template_sha256":"different_source"}
        with patch("compare_transient_runs._load",side_effect=[first,tuple(changed)]):
            with self.assertRaisesRegex(ValueError,"hashes differ"):
                compare_runs("synthetic_A","synthetic_B")
        changed=list(copy.deepcopy(second));changed[3][-1][0]=2
        with patch("compare_transient_runs._load",side_effect=[first,tuple(changed)]):
            with self.assertRaisesRegex(ValueError,"histories differ"):
                compare_runs("synthetic_A","synthetic_B")

    def test_velocity_requires_explicit_tolerance_and_reports_direction(self):
        ref=[(0.,0.,0.,0.),(1.,1.,0.,0.)]
        can=[(0.,0.,0.,0.),(1.,-1.,0.,0.)]
        out=compare_speed(ref,can,atol_m_s=0,rtol=0)
        self.assertTrue(out["speed_tolerance_passed"])
        self.assertEqual(out["maximum_speed_difference_m_s"],0)
        self.assertEqual(out["maximum_velocity_vector_difference_m_s"],2)
        with self.assertRaises(ValueError):compare_speed(ref,can,atol_m_s=-1,rtol=0)
        with self.assertRaises(ValueError):compare_speed(ref,[(0.,0.,0.,0.),(2.,1.,0.,0.)],atol_m_s=0,rtol=0)

    def test_near_rest_uses_absolute_tolerance(self):
        ref=[(0.,0.,0.,0.),(1.,0.,0.,0.)]
        can=[(0.,0.,0.,0.),(1.,.0002,0.,0.)]
        self.assertFalse(compare_speed(ref,can,atol_m_s=.0001,rtol=.01)["speed_tolerance_passed"])
        self.assertTrue(compare_speed(ref,can,atol_m_s=.0003,rtol=0)["speed_tolerance_passed"])


if __name__=="__main__":unittest.main()
