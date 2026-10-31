"""Compare two actual dynamics velocity histories with explicit speed tolerances.

CSV columns: t_s,vx_m_s,vy_m_s (optional vz_m_s). Both runs must start from the
same mechanical conditions and differ only in the chosen magnetic model.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path


def read_velocity_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if not {"t_s","vx_m_s","vy_m_s"}<=set(reader.fieldnames or ()):
            raise ValueError("Velocity CSV requires t_s,vx_m_s,vy_m_s")
        rows=[(float(r["t_s"]),float(r["vx_m_s"]),float(r["vy_m_s"]),float(r.get("vz_m_s",0))) for r in reader]
    if len(rows)<2 or not all(math.isfinite(v) for r in rows for v in r):
        raise ValueError("At least two finite velocity rows are required")
    if any(b[0]<=a[0] for a,b in zip(rows,rows[1:])):
        raise ValueError("Velocity times must be strictly increasing")
    return rows


def compare_speed(reference,candidate,*,atol_m_s,rtol):
    if not all(math.isfinite(x) and x>=0 for x in (atol_m_s,rtol)):
        raise ValueError("Supply nonnegative finite absolute and relative speed tolerances")
    ta=[r[0] for r in reference];tb=[r[0] for r in candidate]
    if not math.isclose(ta[0],tb[0],abs_tol=1e-12) or not math.isclose(ta[-1],tb[-1],abs_tol=1e-12):
        raise ValueError("Compare complete runs covering the same time interval")
    times=sorted(set(ta+tb))
    if times[-1]<=times[0]:raise ValueError("Nonzero duration required")
    def interp(rows,ts,t):
        j=bisect.bisect_left(ts,t)
        if j==0:return rows[0][1:]
        if j==len(ts):return rows[-1][1:]
        u=(t-ts[j-1])/(ts[j]-ts[j-1])
        return tuple(a+u*(b-a) for a,b in zip(rows[j-1][1:],rows[j][1:]))
    norm=lambda v:math.sqrt(sum(x*x for x in v))
    vr=[interp(reference,ta,t) for t in times];vc=[interp(candidate,tb,t) for t in times]
    sr=list(map(norm,vr));sc=list(map(norm,vc))
    speed_delta=[abs(a-b) for a,b in zip(sr,sc)]
    vector_delta=[norm(tuple(b-a for a,b in zip(x,y))) for x,y in zip(vr,vc)]
    scale=max(sr)
    allowed=atol_m_s+rtol*scale
    max_error=max(speed_delta)
    duration=times[-1]-times[0]
    rms=math.sqrt(sum((b-a)*(x*x+y*y)/2 for a,b,x,y in zip(times,times[1:],speed_delta,speed_delta[1:]))/duration)
    return {"maximum_speed_difference_m_s":max_error,"rms_speed_difference_m_s":rms,
            "maximum_velocity_vector_difference_m_s":max(vector_delta),
            "reference_peak_speed_m_s":scale,"absolute_tolerance_m_s":atol_m_s,
            "relative_tolerance":rtol,"allowed_maximum_speed_difference_m_s":allowed,
            "speed_tolerance_passed":max_error<=allowed,
            "sample_count":len(times),"time_end_s":times[-1],
            "scope":"Speed metric on supplied time grids with linear velocity interpolation; no certification of unresolved events.",
            "requires_separate_check":["Same initial mechanical conditions and numerical controls for both runs.",
                                       "Start/stop, slip/roll and direction changes, especially near friction thresholds.",
                                       "Time convergence and material-parameter uncertainty."]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path)
    p.add_argument("candidate",type=Path)
    p.add_argument("--atol",type=float,required=True,help="Absolute speed tolerance in m/s")
    p.add_argument("--rtol",type=float,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=compare_speed(read_velocity_csv(a.reference),read_velocity_csv(a.candidate),atol_m_s=a.atol,rtol=a.rtol)
    result.update(reference_file=str(a.reference.resolve()),candidate_file=str(a.candidate.resolve()))
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
