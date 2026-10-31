"""Verify extracted static interfaces and every archived audited raw point."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import zipfile


HERE=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="mh_") as temporary:
    stage=Path(temporary).resolve()
    with zipfile.ZipFile(HERE/"01_动力学接口_源码与数据.zip") as archive:
        archive.extractall(stage)
    tests=subprocess.run([sys.executable,"-B","-X","utf8","-m","unittest","test_shared_parametric_map","-q"],
                         cwd=stage/"连续磁力地图",capture_output=True,text=True,encoding="utf-8")
    if tests.returncode:raise RuntimeError(tests.stderr)
    audit_code=r'''
import json,math,sys
from pathlib import Path
root=Path.cwd()
sys.path.insert(0,str(root/'连续磁力地图'))
sys.path.insert(0,str(root/'静磁力地图'/'全状态补算_20261029'))
from shared_parametric_map import SharedStaticForceMap,ETAS,FIELD_KEYS
from aggregate_campaign import parse_table
from shared_parameter_campaign import load_trusted_samples
maps=SharedStaticForceMap()
f=maps.query_lab(material='H62',N=4,R_mm=41,g_mm=4,D_mm=6,x_mm=0,y_mm=41,rotor_angle_rad=0)
assert all(f[key]==0 for key in ('Fx_N','Fy_N','Fz_N'))
blocked=False
try:maps.query_lab(material='PC40',N=4,R_mm=41,g_mm=4,D_mm=6,x_mm=0,y_mm=41,rotor_angle_rad=0)
except RuntimeError as error:blocked='not validated' in str(error)
assert blocked
campaign=root/'静磁力地图'/'全状态补算_20261029'
snapshot=json.loads((root/'静磁有效样本快照.json').read_text(encoding='utf-8'))
count=0
for gid,points in snapshot.items():
    for point in points:
        _,forces=parse_table(campaign/point['source_force'])
        _,fields=parse_table(campaign/point['source_field'])
        assert len(forces)==1 and len(fields)==11
        by_eta={round(row['eta'],8):row for row in fields}
        assert sorted(by_eta)==list(ETAS)
        vector=[forces[0][key] for key in ('Fx_N','Fy_N','Fz_N')]
        if gid.startswith('H62_'):vector=[0.,0.,0.]
        vector += [by_eta[eta][key] for eta in ETAS for key in FIELD_KEYS]
        assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(vector,point['values']))
        count+=1
trusted=load_trusted_samples(campaign)
print(json.dumps({'raw_points_reconstructed_from_archived_tables':count,
                  'trusted_loader_point_count':sum(len(v) for v in trusted.values()),
                  'H62_static_zero_query_passed':True,
                  'unvalidated_PC40_query_blocked':blocked,
                  'native_COMSOL_started':False,'numerical_certification':False}))
'''
    audit=subprocess.run([sys.executable,"-B","-X","utf8","-c",audit_code],cwd=stage,
                         capture_output=True,text=True,encoding="utf-8")
    if audit.returncode:raise RuntimeError(audit.stderr)
    result={"static_software_tests_returncode":tests.returncode,"static_software_tests_output":tests.stderr,
            "archive_data_check":json.loads(audit.stdout),"verified_from_actual_extracted_package":True}
    (HERE/"静磁归档与接口验证.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
