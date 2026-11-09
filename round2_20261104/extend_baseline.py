"""Repeat original baseline too, to isolate stem gain from attention gain."""
import json,os,time,subprocess,sys,csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score
from data import prepare,Images
from models import make_model

root=Path(__file__).resolve().parent
status={'state':'waiting','current':'candidate_control_repeats'}
def save():
    temp=root/'logs/extension_status.tmp';temp.write_text(json.dumps(status,indent=2))
    temp.replace(root/'logs/extension_status.json')
save()
try:
    while True:
        main=json.loads((root/'logs/status.json').read_text())
        if main['state']=='failed':raise RuntimeError('Main controller failed')
        if main['state']=='completed':break
        time.sleep(5)
    status['state']='running';save()
    for seed in [43,44]:
        status['current']={'variant':'original','seed':seed};save()
        cmd=[sys.executable,'-u','train.py','--variant','original','--seed',str(seed),
            '--epochs','60','--data','../data','--output','results','--weight-decay','0.0001']
        with (root/'logs/commands.jsonl').open('a') as f:f.write(json.dumps(cmd)+'\n')
        subprocess.run(cmd,cwd=root,check=True)
    status['current']='baseline_test_evaluation';save()
    torch.set_num_threads(4)
    split=prepare('../data')
    metrics=json.loads((root/'metrics.json').read_text())
    with (root/'original_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['variant','seed','condition','path','label','prediction']);w.writeheader()
        for seed in [42,43,44]:
            run=root/f'results/original_seed{seed}'
            ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            model=make_model('original').cuda().eval();model.load_state_dict(ckpt['state_dict'])
            stats=json.loads((run/'complexity.json').read_text())
            for condition in ['clean','dark','blur']:
                ds=Images('../data',split,'test',condition)
                dl=DataLoader(ds,batch_size=128,num_workers=4,pin_memory=True)
                labels=[];preds=[]
                with torch.inference_mode():
                    for x,y in dl:
                        preds.extend(model(x.cuda(non_blocking=True)).argmax(1).cpu().tolist());labels.extend(y.tolist())
                row={'variant':'original','seed':seed,'condition':condition,'accuracy':accuracy_score(labels,preds),
                    'macro_f1':f1_score(labels,preds,average='macro'),'correct':sum(a==b for a,b in zip(labels,preds)),
                    'test_count':len(labels),'best_epoch':ckpt['epoch'],'val_accuracy':ckpt['val_accuracy'],**stats}
                metrics.append(row);print('EVALUATION',json.dumps(row),flush=True)
                for i,(a,b) in enumerate(zip(labels,preds)):
                    w.writerow({'variant':'original','seed':seed,'condition':condition,
                        'path':split['rows'][split['test'][i]]['path'],'label':a,'prediction':b})
                f.flush()
            del model;torch.cuda.empty_cache()
    (root/'metrics_all.json').write_text(json.dumps(metrics,indent=2))
    status.update(state='completed',current='finished',finished_unix=time.time());save()
except BaseException as e:
    status.update(state='failed',error=repr(e));save();raise
