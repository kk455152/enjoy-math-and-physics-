"""Parameterised background-field model for an alternating cuboid magnet ring.

The field is evaluated in air from the magnetic surface-charge integral of
uniformly magnetised cuboids. It does not include a steel ball's reaction field,
eddy currents, hysteresis, or magnetic force.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

MU0 = 4.0e-7 * math.pi
DEFAULT_RADIAL_LENGTH_M = 0.030
DEFAULT_TANGENTIAL_WIDTH_M = 0.010
DEFAULT_AXIAL_THICKNESS_M = 0.003
DEFAULT_BR_T = 1.42
ALLOWED_N = (2, 4, 6, 8, 10)
R_MIN_M = 0.041
R_MAX_M = 0.096
G_MIN_M = 0.002


@dataclass(frozen=True)
class MagnetRing:
    count: int
    radius_m: float
    remanence_T: float = DEFAULT_BR_T
    radial_length_m: float = DEFAULT_RADIAL_LENGTH_M
    tangential_width_m: float = DEFAULT_TANGENTIAL_WIDTH_M
    axial_thickness_m: float = DEFAULT_AXIAL_THICKNESS_M
    quadrature_order: int = 24  # retained for input compatibility; exact face formula is used

    def validate(self) -> None:
        if self.count not in ALLOWED_N:
            raise ValueError(f"N must be one of {ALLOWED_N}.")
        if not math.isfinite(self.radius_m) or not R_MIN_M <= self.radius_m <= R_MAX_M:
            raise ValueError("R must be in the inclusive range 41-96 mm.")
        if not math.isfinite(self.remanence_T) or self.remanence_T <= 0:
            raise ValueError("Br must be a positive finite value in tesla.")
        if self.radial_length_m <= 0 or self.tangential_width_m <= 0:
            raise ValueError("Magnet face dimensions must be positive.")
        if self.axial_thickness_m <= 0:
            raise ValueError("Magnet thickness must be positive.")
        if self.quadrature_order < 8:
            raise ValueError("quadrature_order must be at least 8.")
        if self.has_overlapping_magnets():
            raise ValueError("The selected R and N make adjacent magnet cuboids overlap.")

    def has_overlapping_magnets(self) -> bool:
        centers = []
        axes = []
        half_lengths = []
        half_widths = []
        for index in range(self.count):
            theta = 2.0 * math.pi * index / self.count
            radial = np.array([math.cos(theta), math.sin(theta)])
            tangential = np.array([-math.sin(theta), math.cos(theta)])
            centers.append(self.radius_m * radial)
            axes.append((radial, tangential))
            half_lengths.append(self.radial_length_m / 2.0)
            half_widths.append(self.tangential_width_m / 2.0)

        for i in range(self.count):
            for j in range(i + 1, self.count):
                delta = centers[j] - centers[i]
                separated = False
                for axis in axes[i] + axes[j]:
                    radius_i = (
                        half_lengths[i] * abs(float(np.dot(axes[i][0], axis)))
                        + half_widths[i] * abs(float(np.dot(axes[i][1], axis)))
                    )
                    radius_j = (
                        half_lengths[j] * abs(float(np.dot(axes[j][0], axis)))
                        + half_widths[j] * abs(float(np.dot(axes[j][1], axis)))
                    )
                    if abs(float(np.dot(delta, axis))) >= radius_i + radius_j:
                        separated = True
                        break
                if not separated:
                    return True
        return False

    def field_T(self, points_m: np.ndarray, phase_rad: float = 0.0) -> np.ndarray:
        """Return the air field B in tesla at an array of global xyz points."""
        self.validate()
        points = np.asarray(points_m, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points_m must have a final dimension of length 3.")
        original_shape = points.shape
        flat = points.reshape((-1, 3))
        if not np.all(np.isfinite(flat)):
            raise ValueError("Observation coordinates must be finite.")

        magnetization_Apm = self.remanence_T / MU0
        total_B = np.zeros_like(flat)

        for index in range(self.count):
            theta = 2.0 * math.pi * index / self.count + phase_rad
            radial = np.array([math.cos(theta), math.sin(theta)])
            tangential = np.array([-math.sin(theta), math.cos(theta)])
            center_xy = self.radius_m * radial
            polarity = 1.0 if index % 2 == 0 else -1.0

            delta_xy = flat[:, :2] - center_xy
            local_x = delta_xy @ radial
            local_y = delta_xy @ tangential
            local_z = flat[:, 2] - self.axial_thickness_m / 2.0
            local_H = _cuboid_surface_charge_H(
                local_x,
                local_y,
                local_z,
                self.radial_length_m / 2.0,
                self.tangential_width_m / 2.0,
                self.axial_thickness_m / 2.0,
                polarity * magnetization_Apm,
            )
            global_H = np.empty_like(local_H)
            global_H[:, 0] = local_H[:, 0] * radial[0] + local_H[:, 1] * tangential[0]
            global_H[:, 1] = local_H[:, 0] * radial[1] + local_H[:, 1] * tangential[1]
            global_H[:, 2] = local_H[:, 2]
            total_B += MU0 * global_H

        return total_B.reshape(original_shape)


def _cuboid_surface_charge_H(
    local_x: np.ndarray,
    local_y: np.ndarray,
    local_z: np.ndarray,
    half_length_m: float,
    half_width_m: float,
    half_thickness_m: float,
    magnetization_Apm: float,
) -> np.ndarray:
    """Field of the two uniformly charged rectangular pole faces.

    The expressions are the closed-form rectangle-face integrals. They remain
    accurate for small clearances where a fixed quadrature grid would converge
    slowly. The logarithms are evaluated with positive distances because the
    supported sphere samples are above the magnet top face.
    """
    x = np.asarray(local_x, dtype=float)
    y = np.asarray(local_y, dtype=float)
    z = np.asarray(local_z, dtype=float)
    points = np.column_stack((x, y, z))
    field = np.zeros_like(points)
    for face_z, surface_charge in (
        (half_thickness_m, magnetization_Apm),
        (-half_thickness_m, -magnetization_Apm),
    ):
        dz = z - face_z
        face_field = np.zeros_like(points)
        for ix, x_edge in enumerate((-half_length_m, half_length_m)):
            for iy, y_edge in enumerate((-half_width_m, half_width_m)):
                dx = x - x_edge
                dy = y - y_edge
                distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                sign = (-1.0) ** (ix + iy)
                log_x_arg = dy + distance
                log_y_arg = dx + distance
                if np.any(log_x_arg <= 0.0) or np.any(log_y_arg <= 0.0):
                    raise ValueError("An observation point lies on a magnetic pole face.")
                face_field[:, 0] += sign * np.log(log_x_arg)
                face_field[:, 1] += sign * np.log(log_y_arg)
                face_field[:, 2] += sign * np.arctan2(dx * dy, dz * distance)
        field += surface_charge * face_field / (4.0 * math.pi)
    return field


def sphere_volume_samples(
    diameter_m: float,
    center_xyz_m: tuple[float, float, float],
    height_layers: int = 21,
    points_across_diameter: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create spatial sample points inside a sphere, grouped by vertical layer."""
    if not math.isfinite(diameter_m) or not 0.003 <= diameter_m <= 0.010:
        raise ValueError("D must be in the inclusive range 3-10 mm.")
    if height_layers < 3:
        raise ValueError("height_layers must be at least 3.")
    if points_across_diameter < 3:
        raise ValueError("points_across_diameter must be at least 3.")
    center = np.asarray(center_xyz_m, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xyz_m must contain three finite coordinates.")

    radius = diameter_m / 2.0
    fractions = np.linspace(-1.0, 1.0, height_layers)
    samples = []
    layer_ids = []
    height_fractions = []
    for layer, eta in enumerate(fractions):
        slice_radius = radius * math.sqrt(max(0.0, 1.0 - eta * eta))
        if slice_radius == 0.0:
            offsets = np.zeros((1, 2))
        else:
            axis = np.linspace(-slice_radius, slice_radius, points_across_diameter)
            xx, yy = np.meshgrid(axis, axis, indexing="xy")
            mask = xx * xx + yy * yy <= slice_radius * slice_radius * (1.0 + 1e-12)
            offsets = np.column_stack((xx[mask], yy[mask]))
        z = center[2] + eta * radius
        layer_points = np.column_stack(
            (
                center[0] + offsets[:, 0],
                center[1] + offsets[:, 1],
                np.full(len(offsets), z),
            )
        )
        samples.append(layer_points)
        layer_ids.extend([layer] * len(layer_points))
        height_fractions.extend([eta] * len(layer_points))
    return (
        np.vstack(samples),
        np.asarray(layer_ids, dtype=int),
        np.asarray(height_fractions, dtype=float),
    )


def center_height_from_gap(
    magnet_thickness_m: float,
    gap_m: float,
    plate_thickness_m: float,
    diameter_m: float,
) -> float:
    """Convert magnet-top-to-plate-bottom clearance into ball-center height."""
    if gap_m <= G_MIN_M or plate_thickness_m < 0:
        raise ValueError("g must be greater than 2 mm and plate thickness nonnegative.")
    if diameter_m <= 0:
        raise ValueError("D must be positive.")
    return magnet_thickness_m + gap_m + plate_thickness_m + diameter_m / 2.0


def height_profile(points_m: np.ndarray, field_T: np.ndarray, layer_ids: np.ndarray) -> list[dict]:
    """Summarise the sampled field on each horizontal sphere slice."""
    field = np.asarray(field_T, dtype=float).reshape((-1, 3))
    points = np.asarray(points_m, dtype=float).reshape((-1, 3))
    layer_ids = np.asarray(layer_ids, dtype=int)
    if len(field) != len(points) or len(field) != len(layer_ids):
        raise ValueError("Field, point, and layer arrays must have equal length.")
    rows = []
    for layer in np.unique(layer_ids):
        mask = layer_ids == layer
        b = field[mask]
        norm = np.linalg.norm(b, axis=1)
        row = {
            "layer": int(layer),
            "z_mean_mm": float(np.mean(points[mask, 2]) * 1000.0),
            "sample_count": int(np.count_nonzero(mask)),
            "Bx_mean_T": float(np.mean(b[:, 0])),
            "By_mean_T": float(np.mean(b[:, 1])),
            "Bz_mean_T": float(np.mean(b[:, 2])),
            "Bnorm_mean_T": float(np.mean(norm)),
            "Bnorm_max_T": float(np.max(norm)),
        }
        rows.append(row)
    return rows


def _write_csv(path: Path, field_rows: Iterable[dict], profile_rows: Iterable[dict]) -> None:
    field_rows = list(field_rows)
    profile_rows = list(profile_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep decimal values in filenames without letting Path treat them as a suffix.
    stem = path.name
    field_path = path.with_name(stem + "_field.csv")
    profile_path = path.with_name(stem + "_height_profile.csv")
    if field_rows:
        with field_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(field_rows[0]))
            writer.writeheader()
            writer.writerows(field_rows)
    if profile_rows:
        with profile_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
            writer.writeheader()
            writer.writerows(profile_rows)


def run_sweep(args: argparse.Namespace) -> list[Path]:
    output = Path(args.output_dir)
    generated = []
    for count, radius_mm, gap_mm, diameter_mm, ball_radius_mm, phase_deg in itertools.product(
        args.N, args.R_mm, args.g_mm, args.D_mm, args.rho_mm, args.phase_deg
    ):
        if not radius_mm - 15.0 <= ball_radius_mm <= radius_mm + 15.0:
            raise ValueError(
                "The ball orbit radius must lie in the magnet radial projection "
                "[R-15 mm, R+15 mm]."
            )
        ring = MagnetRing(
            count=count,
            radius_m=radius_mm / 1000.0,
            quadrature_order=args.quadrature_order,
        )
        ring.validate()
        diameter_m = diameter_mm / 1000.0
        if args.z_center_mm is None:
            if args.plate_thickness_mm is None:
                raise ValueError(
                    "Provide --plate-thickness-mm to derive z_center from g, "
                    "or provide --z-center-mm explicitly."
                )
            z_center_m = center_height_from_gap(
                ring.axial_thickness_m,
                gap_mm / 1000.0,
                args.plate_thickness_mm / 1000.0,
                diameter_m,
            )
        else:
            z_center_m = args.z_center_mm / 1000.0

        # The target magnet is at azimuth zero when phase is zero.
        center_xyz_m = (
            ball_radius_mm / 1000.0,
            0.0,
            z_center_m,
        )
        points_m, layer_ids, etas = sphere_volume_samples(
            diameter_m,
            center_xyz_m,
            height_layers=args.height_layers,
            points_across_diameter=args.points_across_diameter,
        )
        if np.min(points_m[:, 2]) <= ring.axial_thickness_m:
            raise ValueError("Sampled sphere points must lie above the magnet top face.")
        field_T = ring.field_T(points_m, phase_rad=math.radians(phase_deg))
        profile_rows = height_profile(points_m, field_T, layer_ids)

        stem = (
            f"N{count}_R{radius_mm:g}mm_g{gap_mm:g}mm_D{diameter_mm:g}mm"
            f"_rho{ball_radius_mm:g}mm_phase{phase_deg:g}deg"
        )
        base_path = output / stem
        field_rows = []
        for point, b, layer, eta in zip(points_m, field_T, layer_ids, etas):
            field_rows.append(
                {
                    "N": count,
                    "R_mm": radius_mm,
                    "g_mm": gap_mm,
                    "D_mm": diameter_mm,
                    "rho_ball_mm": ball_radius_mm,
                    "phase_deg": phase_deg,
                    "Omega_rad_per_s": args.omega_rad_s,
                    "time_s": (
                        math.radians(phase_deg) / args.omega_rad_s
                        if args.omega_rad_s not in (None, 0.0)
                        else None
                    ),
                    "u_relative_mm": (
                        ball_radius_mm * math.cos(math.radians(phase_deg)) - radius_mm
                    ),
                    "v_relative_mm": (
                        -ball_radius_mm * math.sin(math.radians(phase_deg))
                    ),
                    "over_target_magnet_radial_projection": (
                        abs(ball_radius_mm * math.cos(math.radians(phase_deg)) - radius_mm)
                        <= ring.radial_length_m * 1000.0 / 2.0
                    ),
                    "over_target_magnet_tangential_projection": (
                        abs(ball_radius_mm * math.sin(math.radians(phase_deg)))
                        <= ring.tangential_width_m * 1000.0 / 2.0
                    ),
                    "z_center_mm": z_center_m * 1000.0,
                    "layer": int(layer),
                    "height_fraction_eta": float(eta),
                    "x_mm": float(point[0] * 1000.0),
                    "y_mm": float(point[1] * 1000.0),
                    "z_mm": float(point[2] * 1000.0),
                    "Bx_T": float(b[0]),
                    "By_T": float(b[1]),
                    "Bz_T": float(b[2]),
                    "Bnorm_T": float(np.linalg.norm(b)),
                    "Hx_Apm": float(b[0] / MU0),
                    "Hy_Apm": float(b[1] / MU0),
                    "Hz_Apm": float(b[2] / MU0),
                }
            )
        _write_csv(base_path, field_rows, profile_rows)
        generated.extend(
            [
                base_path.with_name(base_path.name + "_field.csv"),
                base_path.with_name(base_path.name + "_height_profile.csv"),
            ]
        )
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute magnet-only B/H fields through spherical sample layers."
    )
    parser.add_argument("--N", type=int, nargs="+", required=True, choices=ALLOWED_N)
    parser.add_argument("--R-mm", type=float, nargs="+", required=True)
    parser.add_argument("--g-mm", type=float, nargs="+", required=True)
    parser.add_argument("--D-mm", type=float, nargs="+", required=True)
    parser.add_argument(
        "--rho-mm",
        type=float,
        nargs="+",
        required=True,
        help="Ball-center orbit radius from the turntable centre.",
    )
    height = parser.add_mutually_exclusive_group()
    height.add_argument("--plate-thickness-mm", type=float)
    height.add_argument("--z-center-mm", type=float)
    parser.add_argument(
        "--phase-deg",
        type=float,
        nargs="+",
        default=[0.0],
        help="Relative drive phase(s); v is derived from this motion phase.",
    )
    parser.add_argument(
        "--omega-rad-s",
        type=float,
        default=None,
        help="Optional drive angular velocity for phase-to-time mapping.",
    )
    parser.add_argument("--height-layers", type=int, default=21)
    parser.add_argument("--points-across-diameter", type=int, default=13)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    if any(not 3.0 <= d <= 10.0 for d in args.D_mm):
        parser.error("Every D value must be between 3 and 10 mm.")
    if any(not 41.0 <= r <= 96.0 for r in args.R_mm):
        parser.error("Every R value must be between 41 and 96 mm.")
    if any(g <= 2.0 for g in args.g_mm):
        parser.error("Every g value must be greater than 2 mm.")
    if any(rho <= 0 for rho in args.rho_mm):
        parser.error("Every ball orbit radius must be positive.")
    if args.omega_rad_s == 0.0:
        parser.error("--omega-rad-s must be nonzero when supplied.")
    if args.plate_thickness_mm is None and args.z_center_mm is None:
        parser.error("Provide --plate-thickness-mm or --z-center-mm.")
    if args.plate_thickness_mm is not None and args.plate_thickness_mm < 0:
        parser.error("Plate thickness must be nonnegative.")
    return args


def main() -> None:
    generated = run_sweep(parse_args())
    print(f"Generated {len(generated)} CSV files:")
    for path in generated:
        print(path.resolve())


if __name__ == "__main__":
    main()
