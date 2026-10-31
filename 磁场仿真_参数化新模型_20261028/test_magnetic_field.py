import math
import unittest

import numpy as np

import magnetic_field as mf
from magnetic_field import ALLOWED_N, MagnetRing, MU0, sphere_volume_samples


class MagneticFieldTests(unittest.TestCase):
    def test_allowed_magnet_counts_are_even_and_bounded(self):
        self.assertEqual(ALLOWED_N, (2, 4, 6, 8, 10))
        for count in ALLOWED_N:
            ring = MagnetRing(count=count, radius_m=0.080)
            ring.validate()

    def test_alternating_ring_has_no_axial_axis_field_by_symmetry(self):
        points = np.array([[0.0, 0.0, 0.025], [0.0, 0.0, 0.060]])
        for count in ALLOWED_N:
            ring = MagnetRing(count=count, radius_m=0.080, quadrature_order=24)
            field = ring.field_T(points)
            # N=2 can retain a transverse field at the ring centre. The
            # alternating axial poles cancel the axial component for every N.
            np.testing.assert_allclose(field[:, 2], 0.0, atol=2e-11, rtol=0.0)

    def test_phase_rotation_is_periodic_over_one_magnet_pitch(self):
        point = np.array([[0.12, 0.007, 0.035]])
        ring = MagnetRing(count=6, radius_m=0.080, quadrature_order=24)
        first = ring.field_T(point, phase_rad=0.37)
        repeated = ring.field_T(point, phase_rad=0.37 + 2.0 * math.pi)
        np.testing.assert_allclose(first, repeated, atol=1e-12, rtol=1e-10)

    def test_ball_sampling_covers_all_layers_and_stays_inside_sphere(self):
        diameter = 0.006
        center = (0.1, 0.0, 0.02)
        points, layers, etas = sphere_volume_samples(
            diameter, center, height_layers=11, points_across_diameter=9
        )
        self.assertEqual(set(layers), set(range(11)))
        radius_sq = diameter * diameter / 4.0
        distance_sq = np.sum((points - np.asarray(center)) ** 2, axis=1)
        self.assertTrue(np.all(distance_sq <= radius_sq * (1.0 + 1e-12)))
        self.assertAlmostEqual(float(etas.min()), -1.0)
        self.assertAlmostEqual(float(etas.max()), 1.0)

    def test_field_scales_linearly_with_remanence(self):
        point = np.array([[0.12, 0.003, 0.025]])
        one = MagnetRing(count=4, radius_m=0.08, remanence_T=1.0).field_T(point)
        two = MagnetRing(count=4, radius_m=0.08, remanence_T=2.0).field_T(point)
        np.testing.assert_allclose(two, 2.0 * one, rtol=1e-10, atol=1e-12)

    def test_rectangular_pole_face_formula_matches_axis_solution(self):
        half_length = 0.015
        half_width = 0.005
        thickness = 0.003
        gap = 0.0001
        magnetization = 1.42 / MU0
        field = mf._cuboid_surface_charge_H(
            np.array([0.0]),
            np.array([0.0]),
            np.array([thickness / 2.0 + gap]),
            half_length,
            half_width,
            thickness / 2.0,
            magnetization,
        )[0, 2]
        corner = lambda z: math.atan(
            half_length * half_width
            / (z * math.sqrt(half_length**2 + half_width**2 + z**2))
        )
        expected = magnetization / math.pi * (corner(gap) - corner(gap + thickness))
        self.assertAlmostEqual(field, expected, places=8)

    def test_out_of_range_radius_is_rejected(self):
        ring = MagnetRing(count=10, radius_m=0.010)
        with self.assertRaisesRegex(ValueError, "41-96"):
            ring.validate()


if __name__ == "__main__":
    unittest.main()
