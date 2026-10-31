"""Shared-parameter static map and reduced per-step coupling entry point.

Running this prints current group readiness. Initialization/trial queries refuse
to replace unavailable FEM data with a different material, geometry or fake force.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"连续磁力地图"))
from shared_parametric_map import SharedStaticForceMap
from stepwise_em_model import StepwiseElectromagneticModel


def build_model(*, material="GCr15_hardened", N=4, R_mm=41, g_mm=4.3, D_mm=5.2):
    maps=SharedStaticForceMap()
    return StepwiseElectromagneticModel(maps,material=material,N=N,R_mm=R_mm,
                                       g_mm=g_mm,D_mm=D_mm)


if __name__=="__main__":
    maps=SharedStaticForceMap()
    print(json.dumps({"method":maps.manifest["policy"],
                      "groups":[{k:g[k] for k in ("group_id","status","force_ready", "centerline_ready")}
                                for g in maps.manifest["groups"]]},ensure_ascii=False,indent=2))
