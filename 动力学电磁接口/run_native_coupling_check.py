"""Actual COMSOL initialization and one transactional moving-ball trial."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
from full_transient_stepper import ComsolTransientStepper
from stepwise_em_model import DynamicState


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="native_coupling_smoke_v10")
    parser.add_argument("--magnetization-model",choices=("jiles_atherton","reversible_bh"),default="jiles_atherton")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/"时变磁场模型"/"通用瞬态耦合"
    settings=json.loads((root/"smoke_GCr15_config.json").read_text(encoding="utf-8"))
    settings.update(mesh_scale=5,dt_s=1e-5,initialization_mode="bh_equilibrated",magnetization_model=args.magnetization_model)
    em=ComsolTransientStepper(settings,root/"results"/args.name)
    s0=DynamicState(0,0,41,0,0,0,0)
    first,memory=em.initialize(s0)
    print("NATIVE_INITIALIZATION_FINISHED",memory.run_directory,flush=True)
    before=memory.history_sha256
    s1=replace(s0,t_s=3e-5,x_mm=.003,vx_m_s=.1,rotor_angle_rad=.0003,
               rotor_speed_rad_s=10,omega_ball_x_rad_s=2,omega_ball_y_rad_s=1,
               omega_ball_z_rad_s=3)
    record,trial=em.trial_step(s1,memory)
    assert memory.history_sha256==before and len(memory.states)==1
    assert trial.states[-1]==s1 and len(trial.states)==2
    summary={"native_initialization_passed":True,"native_trial_passed":True,
             "committed_history_unchanged":True,"initial_record":first,
             "trial_record":record,"trial_memory_json":str(Path(trial.run_directory)/"coupling_memory.json"),
             "numerically_validated":False}
    path=root/"results"/args.name/"interface_check.json"
    path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(path,flush=True)
