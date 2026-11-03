import argparse
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import Images, prepare, seed_worker
from models import make_model, complexity


def run(args):
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = prepare(args.data)
    out = Path(args.output) / f'{args.variant}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({'device': str(device), 'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'initialization': 'random; no pretrained weights', 'input_size': 64})
    (out/'config.json').write_text(json.dumps(config, indent=2))
    trainset, valset = Images(args.data, split, 'train'), Images(args.data, split, 'val')
    if args.smoke:
        trainset = torch.utils.data.Subset(trainset, range(128))
        valset = torch.utils.data.Subset(valset, range(128))
    gen = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i==0), num_workers=args.workers,
        pin_memory=device.type=='cuda', worker_init_fn=seed_worker, generator=gen if i==0 else None,
        persistent_workers=args.workers>0) for i,ds in enumerate([trainset,valset])]
    model = make_model(args.variant).to(device)
    (out/'complexity.json').write_text(json.dumps(complexity(model),indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    best = -1.; start = time.perf_counter()
    with (out/'history.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['epoch','train_loss','train_accuracy','val_loss','val_accuracy','lr','seconds'])
        writer.writeheader()
        for epoch in range(1,args.epochs+1):
            tick=time.perf_counter(); metrics={}
            for mode, loader in zip(['train','val'],loaders):
                model.train(mode=='train'); loss_sum=correct=count=0
                for x,y in loader:
                    x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
                    if mode=='train': optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(mode=='train'):
                        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                            logits=model(x); loss=criterion(logits,y)
                        if mode=='train':
                            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    loss_sum += loss.item()*len(y); correct += (logits.argmax(1)==y).sum().item(); count+=len(y)
                metrics[mode+'_loss']=loss_sum/count
                metrics[mode+'_accuracy']=correct/count
            row={'epoch':epoch,**metrics,'lr':optimizer.param_groups[0]['lr'],'seconds':time.perf_counter()-tick}
            writer.writerow(row); f.flush()
            if metrics['val_accuracy']>best:
                best=metrics['val_accuracy']
                torch.save({'state_dict':model.state_dict(),'variant':args.variant,'seed':args.seed,
                            'epoch':epoch,'val_accuracy':best,'classes':split['classes']},out/'best.pt')
            scheduler.step()
            print(json.dumps({'variant':args.variant,'seed':args.seed,**row}),flush=True)
    (out/'training_summary.json').write_text(json.dumps({'best_val_accuracy':best,
        'total_seconds':time.perf_counter()-start,'completed_epochs':args.epochs},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['baseline','se','cbam'],default='baseline')
    p.add_argument('--data',default='data'); p.add_argument('--output',default='results')
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.001); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
