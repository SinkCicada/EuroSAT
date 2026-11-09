import csv,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score
from models import make_model,complexity
from data import prepare,Images

ROOT=Path(__file__).resolve().parent
DATA='../data'
RESULTS=ROOT/'results'
LOGS=ROOT/'logs'
LOGS.mkdir(exist_ok=True)
status={'state':'running','phase':'smoke','started_unix':time.time(),'completed':[]}
def save():
    temp=LOGS/'status.tmp'
    temp.write_text(json.dumps(status,indent=2));temp.replace(LOGS/'status.json')
def train(variant,seed,wd=1e-4,output='results'):
    status.update(current={'variant':variant,'seed':seed,'weight_decay':wd});save()
    cmd=[sys.executable,'-u','train.py','--variant',variant,'--seed',str(seed),
         '--epochs','60','--data',DATA,'--output',output,'--weight-decay',str(wd)]
    with (LOGS/'commands.jsonl').open('a') as f:f.write(json.dumps(cmd)+'\n')
    print('COMMAND',json.dumps(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    status['completed'].append({'variant':variant,'seed':seed,'weight_decay':wd,'output':output});save()
def val(variant,seed=42,output='results'):
    r=ROOT/output/f'{variant}_seed{seed}'
    return json.loads((r/'training_summary.json').read_text())['best_val_accuracy']

try:
    torch.set_num_threads(4)
    # Verify identical shared backbone initialization and attention gradients.
    states={}
    for variant in ['original','small','small_se','small_cbam']:
        torch.manual_seed(42)
        model=make_model(variant).cuda()
        states[variant]={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'attention' not in k}
        x=torch.randn(2,3,64,64,device='cuda')
        y=model(x);y.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if variant in ['small_se','small_cbam']:
            gradients=[p.grad.abs().sum().item() for n,p in model.named_parameters() if 'attention' in n]
            assert all(g>0 for g in gradients)
        print('SMOKE',variant,list(y.shape),complexity(model),flush=True)
        del model,x,y
        torch.cuda.empty_cache()
    for variant in ['small_se','small_cbam']:
        assert all(torch.equal(v,states[variant][k]) for k,v in states['small'].items())
    del states
    status['phase']='screen_seed42';save()
    for variant in ['original','small','small_se','small_cbam']:train(variant,42)
    screen={v:val(v) for v in ['original','small','small_se','small_cbam']}
    # Predeclared decision: highest validation score; ties prefer simpler model.
    winner=max(screen,key=screen.get)
    if winner in ['small_se','small_cbam']:control='small'
    elif winner=='small':control='original'
    else:control='small'
    selection={'screen_validation':screen,'candidate':winner,'control':control,
               'selection_uses_test':False,'weight_decay':1e-4}
    (ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
    status.update(phase='repeat_seeds43_44',selection=selection);save()
    for seed in [43,44]:
        for variant in [control,winner]:train(variant,seed)
    # Only final, frozen candidate/control pair is evaluated on the existing test.
    status['phase']='final_evaluation';save()
    split=prepare(DATA)
    rows=[]
    with (ROOT/'predictions.csv').open('w',newline='') as pf:
        writer=csv.DictWriter(pf,fieldnames=['variant','seed','condition','path','label','prediction']);writer.writeheader()
        for seed in [42,43,44]:
            for variant in [control,winner]:
                run=RESULTS/f'{variant}_seed{seed}'
                ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
                model=make_model(variant).cuda().eval();model.load_state_dict(ckpt['state_dict'])
                stats=json.loads((run/'complexity.json').read_text())
                for condition in ['clean','dark','blur']:
                    ds=Images(DATA,split,'test',condition)
                    loader=DataLoader(ds,batch_size=128,num_workers=4,pin_memory=True)
                    labels=[];preds=[]
                    with torch.inference_mode():
                        for x,y in loader:
                            pred=model(x.cuda(non_blocking=True)).argmax(1).cpu().tolist()
                            labels.extend(y.tolist());preds.extend(pred)
                    row={'variant':variant,'seed':seed,'condition':condition,'accuracy':accuracy_score(labels,preds),
                         'macro_f1':f1_score(labels,preds,average='macro'),'correct':sum(a==b for a,b in zip(labels,preds)),
                         'test_count':len(labels),'best_epoch':ckpt['epoch'],'val_accuracy':ckpt['val_accuracy'],**stats}
                    rows.append(row)
                    for idx,(a,b) in enumerate(zip(labels,preds)):
                        writer.writerow({'variant':variant,'seed':seed,'condition':condition,
                            'path':split['rows'][split['test'][idx]]['path'],'label':a,'prediction':b})
                    pf.flush();print('EVALUATION',json.dumps(row),flush=True)
                del model
                torch.cuda.empty_cache()
    (ROOT/'metrics.json').write_text(json.dumps(rows,indent=2))
    status.update(state='completed',phase='finished',finished_unix=time.time());save()
except BaseException as e:
    status.update(state='failed',error=repr(e));save();raise
