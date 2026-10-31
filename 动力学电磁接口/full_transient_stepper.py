"""History-preserving COMSOL reference callback for partitioned dynamics.

Each trial replays the complete accepted trajectory plus one trial segment.
This is expensive, but preserves spatial eddy-current and JA history without
pretending that a mean magnetization or a drag law is the full FE state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Callable

from stepwise_em_model import DynamicState

BACKEND_DIR = Path(__file__).resolve().parents[1]/"时变磁场模型"/"通用瞬态耦合"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from comsol_backend import run_forward, read_force_history, atomic_json, validate

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
MotionRow = tuple[float, ...]


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _source_digest() -> str:
    return hashlib.sha256((BACKEND_DIR/"FullTransientCarousel.java").read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _state_ok(s: DynamicState):
    if not isinstance(s, DynamicState) or not all(math.isfinite(v) for v in asdict(s).values()):
        raise ValueError("A finite DynamicState is required")


def _multiply(a: Quaternion,b: Quaternion) -> Quaternion:
    w,x,y,z=a;v,i,j,k=b
    return (w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v)


def _spin(q: Quaternion, omega: Vector3, dt: float) -> Quaternion:
    norm=math.sqrt(sum(v*v for v in omega))
    if norm==0:
        return q
    angle=norm*dt
    factor=math.sin(angle/2)/norm
    inc=(math.cos(angle/2),*(factor*v for v in omega))
    value=_multiply(inc,q)  # supplied angular velocity is in laboratory axes
    length=math.sqrt(sum(v*v for v in value))
    return tuple(v/length for v in value)


def _hermite(a,b,da,db,u,dt):
    return (2*u**3-3*u*u+1)*a+(u**3-2*u*u+u)*dt*da+(-2*u**3+3*u*u)*b+(u**3-u*u)*dt*db


@dataclass(frozen=True)
class FullTransientMemory:
    """Immutable committed or trial history; .mph contains the native FE state."""
    states: tuple[DynamicState, ...]
    motion: tuple[MotionRow, ...]
    settings_sha256: str
    source_sha256: str
    run_directory: str
    checkpoint_mph: str
    force_xyz_N: Vector3
    torque_xyz_Nm: Vector3
    history_sha256: str

    @property
    def t_s(self):
        return self.states[-1].t_s


class ComsolTransientStepper:
    """Common parameterized backend; accepting a step means retaining its memory.

    No automatic commit occurs. A rejected/corrected trial always starts from
    the unchanged supplied memory. .mph continuation is not yet used for speed;
    the entire history is reconstructed in each fresh COMSOL solve.
    """
    def __init__(self,settings: dict, output_root, *, max_motion_dt_s=None,
                 max_translation_mm=None,max_rotation_step_rad=.01,
                 timeout_s=5400., runner: Callable=run_forward):
        self._settings_json=json.dumps(dict(settings),sort_keys=True,allow_nan=False)
        self.output_root=Path(output_root).resolve()
        self.max_motion_dt_s=float(max_motion_dt_s or settings["dt_s"])
        self.max_translation_mm=float(max_translation_mm or float(settings["D_mm"])/12)
        self.max_rotation_step_rad=float(max_rotation_step_rad)
        self.timeout_s=float(timeout_s)
        if not all(math.isfinite(v) and v>0 for v in (
                self.max_motion_dt_s,self.max_translation_mm,self.max_rotation_step_rad,self.timeout_s)):
            raise ValueError("Positive finite motion-resolution controls are required")
        self.runner=runner
        self._identity=_digest({"settings":settings,"max_motion_dt_s":self.max_motion_dt_s,
                              "max_translation_mm":self.max_translation_mm,
                              "max_rotation_step_rad":self.max_rotation_step_rad,
                              "motion_rule":"Hermite_xy_phase_midpoint_lab_omega_v1"})

    @property
    def settings(self):
        return json.loads(self._settings_json)

    def _configuration(self,s: DynamicState):
        settings=self.settings
        settings.update(rho_mm=math.hypot(s.x_mm,s.y_mm),
                        delta_deg=math.degrees(math.atan2(s.y_mm,s.x_mm)-math.pi/2-s.rotor_angle_rad),
                        rotor_initial_rad=s.rotor_angle_rad)
        return settings

    def _run(self,settings,motion,*,stationary=False):
        source=_source_digest()
        key=_digest({"source":source,"settings":settings,"motion":motion,"stationary":stationary})
        self.output_root.mkdir(parents=True,exist_ok=True)
        output=self.output_root/key[:24]
        if output.exists():
            status_file=output/"run_status.json"
            status=json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
            if status.get("status") not in ("solved","stationary_solved") or status.get("template_sha256")!=source:
                raise RuntimeError(f"Previous run incomplete; inspect/remove only that run, or choose a new output_root: {output}")
        else:
            self.runner(settings,motion,output,initialize_only=stationary,timeout_s=self.timeout_s)
        # A source change during a run invalidates the assumed replay identity.
        if _source_digest()!=source:
            raise RuntimeError("COMSOL source changed during the solve; recreate the stepper history")
        rows=read_force_history(output/"total_force_history.txt")
        expected=0 if stationary else motion[-1][0]
        if not rows or not math.isclose(rows[-1]["t_s"],expected,abs_tol=1e-12,rel_tol=1e-9):
            raise RuntimeError("Incomplete COMSOL endpoint")
        if any(row["min_relative_mesh_volume"]<=0 for row in rows):
            raise RuntimeError("Inverted moving mesh; returned fields must not enter dynamics")
        return output,rows[-1],source

    def _record(self,output,row,external_time):
        mode=self.settings.get("magnetization_model","jiles_atherton")
        magnetic=self.settings["material"]!="H62"
        record={"t_s":external_time,
                "Fmagnetic_xyz_N":tuple(row[f"F{a}_total_N"] for a in "xyz"),
                "Tmagnetic_xyz_Nm":tuple(row[f"T{a}_total_Nm"] for a in "xyz"),
                "JxB_integral_xyz_N":tuple(row[f"JxB_integral_{a}_N"] for a in "xyz"),
                "Jrms_A_m2":row["Jrms_A_m2"],"Pjoule_W":row["Pjoule_W"],
                "Mmean_xyz_A_m":tuple(row[f"Mmean_{a}_A_m"] for a in "xyz"),
                "min_relative_mesh_volume":row["min_relative_mesh_volume"],
                "ball_fields_last_csv":str(output/"ball_fields_last.csv"),
                "run_directory":str(output),
                "backend":"COMSOL_spatial_Maxwell_vector_JA_full_causal_replay" if mode=="jiles_atherton" else "COMSOL_spatial_Maxwell_reversible_BH_full_causal_replay",
                "numerically_validated":False,
                "material_parameters":"JA_engineering_estimates_not_measured_loop_fit" if magnetic and mode=="jiles_atherton" else "nonmagnetic_mu_r_1" if not magnetic else "reversible_BH_approximation",
                "magnetization_model":mode if magnetic else "nonmagnetic",
                "bh_curve_source":self.settings.get("bh_curve_source","supplied_table"),
                "hysteresis_retained":magnetic and mode=="jiles_atherton",
                "transient_eddy_currents_retained":True}
        record["force_evaluation"]="volume_JxB_nonmagnetic" if self.settings["material"]=="H62" else "Maxwell_surface"
        if "FMaxwell_x_N" in row:
            record["FMaxwell_surface_xyz_N"]=tuple(row[f"FMaxwell_{a}_N"] for a in "xyz")
        return record

    def initialize(self,state: DynamicState):
        """Return (initial force record, memory) from native stationary init."""
        _state_ok(state)
        settings=self._configuration(state)
        first=(0.,0.,0.,state.rotor_angle_rad,1.,0.,0.,0.)
        motion=(first,(self.max_motion_dt_s,*first[1:]))
        validate(settings,motion)
        output,row,source=self._run(settings,motion,stationary=True)
        record=self._record(output,row,state.t_s)
        history=(first,)
        memory=self._memory((state,),history,output,record,source)
        self._save(memory)
        return record,memory

    def _memory(self,states,motion,output,record,source):
        return FullTransientMemory(states,motion,self._identity,source,str(output),
                str(output/("TransientCarousel_"+self.settings["material"]+"_solved.mph")),
                record["Fmagnetic_xyz_N"],record["Tmagnetic_xyz_Nm"],
                _digest({"states":[asdict(s) for s in states],"motion":motion,
                         "settings":self._identity,"source":source}))

    def _save(self,memory):
        atomic_json(Path(memory.run_directory)/"coupling_memory.json",asdict(memory))

    def _segment(self,start,end,previous,origin):
        dt=end.t_s-start.t_s
        ws=(start.omega_ball_x_rad_s,start.omega_ball_y_rad_s,start.omega_ball_z_rad_s)
        we=(end.omega_ball_x_rad_s,end.omega_ball_y_rad_s,end.omega_ball_z_rad_s)
        # Hermite speed can exceed endpoint speeds; conservative subdivision bounds.
        distance=math.hypot(end.x_mm-start.x_mm,end.y_mm-start.y_mm)
        speed=max(math.hypot(start.vx_m_s,start.vy_m_s),math.hypot(end.vx_m_s,end.vy_m_s))*1000
        spin=max(math.sqrt(sum(v*v for v in w)) for w in (ws,we))
        phase=max(abs(end.rotor_angle_rad-start.rotor_angle_rad),
                  dt*max(abs(start.rotor_speed_rad_s),abs(end.rotor_speed_rad_s)))
        count=max(1,math.ceil(dt/self.max_motion_dt_s),
                  math.ceil(4*max(distance,speed*dt)/self.max_translation_mm),
                  math.ceil(max(spin*dt,4*phase)/self.max_rotation_step_rad))
        if count>200000:
            raise ValueError("Motion segment needs >200000 rows; reduce dynamics dt")
        q=tuple(previous[4:8]);rows=[]
        for i in range(1,count+1):
            u=i/count;mid=(i-.5)/count
            omega=tuple(ws[j]+mid*(we[j]-ws[j]) for j in range(3))
            q=_spin(q,omega,dt/count)
            x=_hermite(start.x_mm,end.x_mm,start.vx_m_s*1000,end.vx_m_s*1000,u,dt)
            y=_hermite(start.y_mm,end.y_mm,start.vy_m_s*1000,end.vy_m_s*1000,u,dt)
            alpha=_hermite(start.rotor_angle_rad,end.rotor_angle_rad,
                           start.rotor_speed_rad_s,end.rotor_speed_rad_s,u,dt)
            rows.append((start.t_s-origin.t_s+dt*u,x-origin.x_mm,y-origin.y_mm,alpha,*q))
        return tuple(rows)

    def trial_step(self,state: DynamicState,memory: FullTransientMemory):
        _state_ok(state)
        if memory.settings_sha256!=self._identity or memory.source_sha256!=_source_digest():
            raise ValueError("Memory belongs to another configuration/source; initialize a new history")
        if state.t_s<=memory.t_s:
            raise ValueError("Trial time must be later than the committed memory")
        expected=_digest({"states":[asdict(s) for s in memory.states],"motion":memory.motion,
                          "settings":self._identity,"source":memory.source_sha256})
        if memory.history_sha256!=expected:
            raise ValueError("Electromagnetic history digest mismatch")
        motion=memory.motion+self._segment(memory.states[-1],state,memory.motion[-1],memory.states[0])
        settings=self._configuration(memory.states[0])
        validate(settings,motion)
        output,row,source=self._run(settings,motion)
        record=self._record(output,row,state.t_s)
        candidate=self._memory(memory.states+(state,),motion,output,record,source)
        self._save(candidate)
        return record,candidate

    def load_memory(self,path):
        """Restore the replay history from JSON; no mean-state reconstruction."""
        value=json.loads(Path(path).read_text(encoding="utf-8"))
        value["states"]=tuple(DynamicState(**s) for s in value["states"])
        value["motion"]=tuple(tuple(r) for r in value["motion"])
        for key in ("force_xyz_N","torque_xyz_Nm"):
            value[key]=tuple(value[key])
        memory=FullTransientMemory(**value)
        if memory.settings_sha256!=self._identity or memory.source_sha256!=_source_digest():
            raise ValueError("Saved memory configuration/source mismatch")
        return memory
