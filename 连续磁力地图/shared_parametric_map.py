"""One four-dimensional static surrogate for each material and magnet layout.

Inputs are orbit-radius offset, normalized relative phase, gap and DIAMETER.
The full annulus is covered; instantaneous tangential position is unrestricted.
The model interpolates reversible FEM data, never magnetic history.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

# Keep scientific dependencies outside the user's transferable data directory.
try:
    from scipy.interpolate import RBFInterpolator
    import numpy as np
except ImportError:
    dependency_dir = Path.home() / ".codex" / "dependencies" / "magnetic_interpolation"
    if dependency_dir.is_dir():
        sys.path.insert(0, str(dependency_dir))
    from scipy.interpolate import RBFInterpolator
    import numpy as np

BOUNDS = ((-15., 15.), (-1., 1.), (3., 6.), (3., 10.))
AXES = ("rho_minus_R_mm", "phase_N_delta_over_pi", "g_mm", "D_mm")
ETAS = tuple(round(-1 + i * .2, 8) for i in range(11))
FIELD_KEYS = ("Bx_T", "By_T", "Bz_T", "Hx_Apm", "Hy_Apm", "Hz_Apm")
MATERIALS = ("H62", "PC40", "GCr15_hardened", "GCr15_annealed")
LAYOUTS = ((4, 41.), (10, 96.))
POLICY = "shared_4d_annulus_periodic_rbf_independent_holdout_v2"


def group_id(material, n, radius):
    if material not in MATERIALS or (n, float(radius)) not in LAYOUTS:
        raise ValueError("Material and layout must match one of the eight requested groups")
    return f"{material}_N{int(n)}_R{float(radius):g}"


def point_key(coordinates):
    return tuple(round(float(x), 6) for x in coordinates)


def equivalent_key(coordinates):
    """The two sector endpoints represent the same state up to known symmetry."""
    c = list(point_key(coordinates))
    if c[1] == 1.:
        c[1] = -1.
    return tuple(c)


def check_coordinates(coordinates):
    if len(coordinates) != 4:
        raise ValueError("Expected rho-R, normalized phase, g, D")
    coordinates = tuple(float(x) for x in coordinates)
    for name, x, (lo, hi) in zip(AXES, coordinates, BOUNDS):
        if not math.isfinite(x) or x < lo - 1e-7 or x > hi + 1e-7:
            raise ValueError(f"{name} outside [{lo:g}, {hi:g}]; extrapolation forbidden")
    return coordinates


def normalize(coordinates):
    """Embed the phase on a circle; the other three inputs are unit-scaled."""
    a = np.asarray(coordinates, dtype=float)
    bounds = np.asarray(BOUNDS)
    unit = (a - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    theta = a[..., 1] * math.pi / 2
    return np.stack((unit[..., 0], .5*np.cos(theta), .5*np.sin(theta),
                     unit[..., 2], unit[..., 3]), axis=-1)


def corners():
    return tuple(itertools.product(*BOUNDS))


class RBFGroup:
    """One periodic four-input interpolant per material/layout.

    Targets are fitted in the ball's orbital cylindrical frame. Paired samples
    enforce periodic force and antiperiodic B/H under one magnet pitch. This
    makes switching to the next magnet continuous, including between magnets.
    """

    def __init__(self, points, material, N=4, *, require_domain_corners=True):
        self.material, self.N = material, N
        self.points = {point_key(p["coordinates"]): p for p in points}
        if len(self.points) < 5:
            raise ValueError("At least five independent parameter points are required")
        if require_domain_corners and any(point_key(c) not in self.points for c in corners()):
            raise ValueError("Full parameter-box corners are missing; coverage not established")
        ordered = sorted(self.points)
        x = np.asarray(ordered)
        values = np.asarray([self.points[c]["values"] for c in ordered], dtype=float)
        if values.shape != (len(x), 69) or not np.isfinite(values).all():
            raise ValueError("Expected three forces and six fields at each of eleven heights")
        for i, c in enumerate(x):
            values[i] = self._frame(values[i], c, inverse=True)
        # Scaling removes leading sphere-volume dependence; finite-size and
        # saturation effects remain in the FEM residual, not an imposed law.
        values[:, :3] /= (x[:, 3:4] / 6.) ** 3
        if material == "H62":
            values[:, :3] = 0.
        features = normalize(x)
        opposite = features.copy()
        opposite[:, 1:3] *= -1
        opposite_values = values.copy()
        opposite_values[:, 3:] *= -1
        # Sector endpoints and their symmetry partners coincide. Average their
        # FEM residuals rather than present duplicate rows to a singular solve.
        combined = {}
        for f, value in zip(np.vstack((features, opposite)),
                            np.vstack((values, opposite_values))):
            combined.setdefault(tuple(np.round(f, 12)), []).append(value)
        features = np.asarray(list(combined))
        if np.linalg.matrix_rank(np.column_stack((np.ones(len(features)), features))) < 6:
            raise ValueError("Phase samples need a sector-interior point in addition to parameter-box corners")
        self.interpolant = RBFInterpolator(
                                           features,
                                           np.asarray([np.mean(v, axis=0) for v in combined.values()]),
                                           kernel="thin_plate_spline", degree=1,
                                           smoothing=0.)

    def _frame(self, values, coordinates, inverse=False):
        angle = math.pi/2 + coordinates[1]*math.pi/self.N
        if inverse:
            angle = -angle
        result = np.array(values, dtype=float, copy=True)
        result[:3] = rotate(result[:3], angle)
        profile = result[3:].reshape(11, 6)
        for row in profile:
            row[:3] = rotate(row[:3], angle)
            row[3:] = rotate(row[3:], angle)
        return result

    def predict(self, coordinates):
        c = check_coordinates(coordinates)
        result = self.interpolant(normalize([c]))[0]
        result[:3] *= (c[3] / 6.) ** 3
        if self.material == "H62":
            result[:3] = 0.
        result = self._frame(result, c)
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite surrogate output")
        return result


def validate(model, training, held_out, expected_count):
    """Held-out COMSOL points are never supplied to RBFGroup's training fit."""
    if len(held_out) != expected_count or not held_out:
        return {"complete": False, "available": len(held_out), "expected": expected_count}
    train_keys = {equivalent_key(p["coordinates"]) for p in training}
    if any(equivalent_key(p["coordinates"]) in train_keys for p in held_out):
        raise ValueError("Validation leakage: held-out point is present in training")
    actual = np.asarray([p["values"] for p in held_out])
    predicted = np.asarray([model.predict(p["coordinates"]) for p in held_out])
    all_values = np.asarray([p["values"] for p in training + held_out])
    if model.material == "H62":
        actual[:, :3] = predicted[:, :3] = all_values[:, :3] = 0.
    delta = predicted - actual
    fscale = max(float(np.linalg.norm(all_values[:, :3], axis=1).max()), 1e-30)
    xyscale = max(float(np.linalg.norm(all_values[:, :2], axis=1).max()), .05 * fscale, 1e-30)
    all_fields = all_values[:, 3:].reshape(-1, 11, 6)
    bscale = max(float(np.linalg.norm(all_fields[:, :, :3], axis=2).max()), 1e-30)
    hscale = max(float(np.linalg.norm(all_fields[:, :, 3:], axis=2).max()), 1e-30)
    field_error = delta[:, 3:].reshape(-1, 11, 6)
    fe = np.linalg.norm(delta[:, :3], axis=1) / fscale
    xy = np.linalg.norm(delta[:, :2], axis=1) / xyscale
    be = np.linalg.norm(field_error[:, :, :3], axis=2).max(axis=1) / bscale
    he = np.linalg.norm(field_error[:, :, 3:], axis=2).max(axis=1) / hscale
    checks = [{"coordinates": p["coordinates"], "force_error_fraction": float(fe[i]),
               "horizontal_error_fraction": float(xy[i]), "B_error_fraction": float(be[i]),
               "H_error_fraction": float(he[i]),
               "score": float(max(fe[i]/.05, xy[i]/.10, be[i]/.05, he[i]/.05))}
              for i, p in enumerate(held_out)]
    return {"complete": True, "training_points": len(training), "holdout_points": len(held_out),
            "force_ready": bool(fe.max() <= .05 and xy.max() <= .10),
            "centerline_ready": bool(be.max() <= .05 and he.max() <= .05),
            "max_force_error_fraction": float(fe.max()),
            "max_horizontal_error_fraction": float(xy.max()),
            "max_B_error_fraction": float(be.max()), "max_H_error_fraction": float(he.max()),
            "force_scale_N": fscale, "horizontal_scale_N": xyscale,
            "B_scale_T": bscale, "H_scale_A_m": hscale,
            "force_tolerance": .05, "horizontal_tolerance": .10, "field_tolerance": .05,
            "checks": checks}


def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return [c*x - s*y, s*x + c*y, z]


class SharedStaticForceMap:
    """Same callable protocol as StaticForceMap, including continuous g and D."""

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parent / "统一参数模型")
        self.refresh()

    def refresh(self):
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("policy") != POLICY:
            raise RuntimeError("Map uses an older rectangular domain; load the annular model")
        self.entries = {g["group_id"]: g for g in self.manifest["groups"]}
        self.cache = {}

    def _model(self, material, N, R_mm, field=False):
        gid = group_id(material, N, R_mm)
        entry = self.entries[gid]
        if not entry["force_ready"] or (field and not entry["centerline_ready"]):
            raise RuntimeError(f"{gid}: shared parameter interpolation is not validated yet")
        if gid not in self.cache:
            payload = json.loads((self.root / entry["file"]).read_text(encoding="utf-8"))
            if payload["template_sha256"] != self.manifest["template_sha256"]:
                raise RuntimeError("Provenance changed; refresh the map")
            if not payload["force_ready"] or (field and not payload["centerline_ready"]):
                raise RuntimeError("Validation state changed; refresh the map")
            if payload.get("policy") != POLICY:
                raise RuntimeError("Parameter domain changed; refresh the map")
            self.cache[gid] = RBFGroup(payload["training"], material, N)
        return self.cache[gid], entry

    def query_local(self, material, N, R_mm, g_mm, D_mm, u_mm, v_mm):
        rho = math.hypot(R_mm + u_mm, v_mm)
        delta = math.atan2(v_mm, R_mm + u_mm)
        coordinates = check_coordinates((rho-R_mm, N*delta/math.pi, g_mm, D_mm))
        if material == "H62":
            group_id(material, N, R_mm)
            force, policy = [0., 0., 0.], "analytical_zero_mu_r_1"
            report = {"force_ready": True, "basis": "nonmagnetic_static_zero"}
        else:
            model, entry = self._model(material, N, R_mm)
            force = model.predict(coordinates)[:3].tolist()
            policy, report = "shared_4d_COMSOL_interpolation", entry["validation"]
        angle = math.pi/2 + math.atan2(v_mm, R_mm + u_mm)
        fx, fy, fz = force
        fr = fx*math.cos(angle) + fy*math.sin(angle)
        ft = -fx*math.sin(angle) + fy*math.cos(angle)
        return {"Fx_N": fx, "Fy_N": fy, "Fz_N": fz,
                "Fr_orbit_N": fr, "Ftheta_orbit_N": ft,
                "Tx_ball_Nm": 0., "Ty_ball_Nm": 0., "Tz_ball_Nm": 0.,
                "Tz_axis_Nm": math.hypot(R_mm+u_mm, v_mm)/1000*ft,
                "force_policy": policy, "model": "reversible_static_shared_4d",
                "validation": report, "g_mm": g_mm, "D_mm": D_mm,
                "rho_mm": rho, "relative_phase_rad": delta,
                "normalized_phase": coordinates[1]}

    @staticmethod
    def _local(N, R_mm, x_mm, y_mm, rotor_angle_rad):
        if not all(math.isfinite(v) for v in (x_mm, y_mm, rotor_angle_rad)):
            raise ValueError("Position and phase must be finite")
        pitch = 2*math.pi/N
        index = math.floor((math.atan2(y_mm, x_mm)-rotor_angle_rad-math.pi/2)/pitch+.5)
        angle = math.pi/2 + rotor_angle_rad + index*pitch
        return (x_mm*math.cos(angle)+y_mm*math.sin(angle)-R_mm,
                -x_mm*math.sin(angle)+y_mm*math.cos(angle), index)

    def query_lab(self, material, N, R_mm, g_mm, D_mm, x_mm, y_mm, rotor_angle_rad):
        group_id(material, N, R_mm)
        u, v, index = self._local(N, R_mm, x_mm, y_mm, rotor_angle_rad)
        result = self.query_local(material, N, R_mm, g_mm, D_mm, u, v)
        force = rotate([result[k] for k in ("Fx_N", "Fy_N", "Fz_N")],
                       rotor_angle_rad + index*2*math.pi/N)
        result.update(dict(zip(("Fx_N", "Fy_N", "Fz_N"), force)))
        result.update(u_mm=u, v_mm=v, magnet_index=index % N,
                      field_polarity=1 if index % 2 == 0 else -1)
        return result

    def centerline_lab(self, material, N, R_mm, g_mm, D_mm, x_mm, y_mm,
                       rotor_angle_rad, eta, allow_unvalidated_z_interpolation=False):
        group_id(material, N, R_mm)
        u, v, index = self._local(N, R_mm, x_mm, y_mm, rotor_angle_rad)
        rho = math.hypot(R_mm+u, v)
        coordinates = check_coordinates((rho-R_mm, N*math.atan2(v, R_mm+u)/math.pi, g_mm, D_mm))
        if not math.isfinite(eta) or eta < -1 or eta > 1:
            raise ValueError("eta must be between -1 and 1")
        exact = next((i for i, e in enumerate(ETAS) if abs(e-eta) < 1e-9), None)
        if exact is None and not allow_unvalidated_z_interpolation:
            raise ValueError("Only eleven eta levels validated; opt in for unvalidated height interpolation")
        model, _ = self._model(material, N, R_mm, field=True)
        profile = model.predict(coordinates)[3:].reshape(11, 6)
        if exact is not None:
            f = profile[exact]
        else:
            lo = min(9, max(0, int(math.floor((eta+1)/.2))))
            fraction = (eta-ETAS[lo])/.2
            f = (1-fraction)*profile[lo] + fraction*profile[lo+1]
        angle = rotor_angle_rad + index*2*math.pi/N
        sign = 1 if index % 2 == 0 else -1
        fields = [sign*x for x in rotate(f[:3], angle) + rotate(f[3:], angle)]
        return {**dict(zip(FIELD_KEYS, fields)), "eta": eta,
                "z_mm": 3+g_mm+1+D_mm*(1+eta)/2,
                "height_interpolation_independently_validated": exact is not None}
