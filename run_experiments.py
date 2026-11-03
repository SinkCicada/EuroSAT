"""Sequential controller; its PID, complete command and output are saved by deploy."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--epochs',type=int,default=30)
p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
a=p.parse_args()
root=Path(__file__).resolve().parent
(root/'logs').mkdir(exist_ok=True)
status={'state':'running','epochs':a.epochs,'seeds':a.seeds,'started_unix':time.time(),'completed':[]}
def write():
    (root/'logs/status.json').write_text(json.dumps(status,indent=2))
def execute(cmd):
    print('COMMAND',json.dumps(cmd),flush=True)
    r=subprocess.run(cmd,cwd=root)
    if r.returncode:
        status.update(state='failed',failed_command=cmd,exit_code=r.returncode);write()
        raise SystemExit(r.returncode)
write()
execute([sys.executable,'smoke.py'])
for seed in a.seeds:
    for variant in ['baseline','se','cbam']:
        status['current']={'seed':seed,'variant':variant};write()
        cmd=[sys.executable,'train.py','--variant',variant,'--seed',str(seed),'--epochs',str(a.epochs)]
        execute(cmd)
        status['completed'].append({'seed':seed,'variant':variant});write()
status['current']='test_evaluation';write()
execute([sys.executable,'analyze.py'])
status.update(state='completed',finished_unix=time.time());write()
