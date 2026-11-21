import sys, json
sys.path.insert(0, '.')
from lerobot_audit.profiles import get_profile
from lerobot_audit.audit import run_audit
r = run_audit(get_profile(sys.argv[1]))
s = r.summary
for k in ['slice','lengths_full_meta','lengths_slice','idle','smoothness','gripper','integrity','duplicates','task_consistency','hygiene','coverage','usable','success','language']:
    print(k, json.dumps(s.get(k), default=str)[:700])
print(r.episodes.drop(columns=['hash','task']).describe(percentiles=[.05,.5,.95]).T.to_string())
