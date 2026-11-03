"""Evaluate held-out data once after training; create figures from saved outputs."""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader

from data import Images, prepare
from models import make_model

COLORS={'baseline':'#32658c','se':'#d18b28','cbam':'#32856c'}


def load_model(path, device):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model=make_model(checkpoint['variant']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(),checkpoint


def gradcam(model, tensor, target=None):
    activation=[]
    def hook(_,inputs,output):
        activation.append(output)
        output.retain_grad()
    handle=model.layer3.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    logits=model(tensor)
    target=logits.argmax(1).item() if target is None else target
    logits[0,target].backward()
    a=activation[0]
    weights=a.grad.mean((2,3),keepdim=True)
    cam=(weights*a).sum(1,keepdim=True).relu()
    cam=torch.nn.functional.interpolate(cam,size=tensor.shape[-2:],mode='bilinear',align_corners=False)[0,0]
    cam=cam.detach().cpu().numpy()
    cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    probs=logits.softmax(1)[0].detach().cpu().numpy()
    handle.remove()
    return cam,probs


def evaluate(args):
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split=prepare(args.data)
    root=Path(args.results); figures=root/'figures';figures.mkdir(exist_ok=True,parents=True)
    rows=[]; predictions=[]
    for run in sorted(root.glob('*_seed*')):
        if not (run/'training_summary.json').exists():continue
        model,ckpt=load_model(run/'best.pt',device)
        stats=json.loads((run/'complexity.json').read_text())
        summary=json.loads((run/'training_summary.json').read_text())
        for condition in ['clean','dark','blur']:
            ds=Images(args.data,split,'test',condition)
            loader=DataLoader(ds,batch_size=128,num_workers=4,pin_memory=device.type=='cuda')
            labels=[];preds=[];confs=[]
            with torch.inference_mode():
                for x,y in loader:
                    prob=model(x.to(device)).softmax(1).cpu()
                    labels.extend(y.tolist());preds.extend(prob.argmax(1).tolist());confs.extend(prob.max(1).values.tolist())
            record={'variant':ckpt['variant'],'seed':ckpt['seed'],'condition':condition,
                'test_count':len(labels),'correct':int(np.sum(np.array(labels)==preds)),
                'accuracy':accuracy_score(labels,preds),'macro_f1':f1_score(labels,preds,average='macro'),
                'best_epoch':ckpt['epoch'],'val_accuracy':ckpt['val_accuracy'],
                'training_seconds':summary['total_seconds'],**stats}
            rows.append(record)
            (run/f'{condition}_classification.json').write_text(json.dumps(
                classification_report(labels,preds,target_names=split['classes'],output_dict=True),indent=2))
            np.save(run/f'{condition}_confusion.npy',confusion_matrix(labels,preds))
            for i,(y,p,c) in enumerate(zip(labels,preds,confs)):
                predictions.append({'variant':ckpt['variant'],'seed':ckpt['seed'],'condition':condition,
                    'path':split['rows'][split['test'][i]]['path'],'label':y,'prediction':p,'confidence':c})
            print(json.dumps(record),flush=True)
        # Same FP32 batch-1 CUDA timing for every model. Shared GPU timings are descriptive.
        x=torch.zeros(1,3,64,64,device=device)
        elapsed=[]
        with torch.inference_mode():
            for _ in range(30):model(x)
            if device.type=='cuda':torch.cuda.synchronize()
            for _ in range(100):
                if device.type=='cuda':
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();model(x);end.record();end.synchronize();elapsed.append(begin.elapsed_time(end))
                else:
                    tick=time.perf_counter();model(x);elapsed.append((time.perf_counter()-tick)*1000)
        (run/'latency.json').write_text(json.dumps({'median_ms':float(np.median(elapsed)),
            'p90_ms':float(np.percentile(elapsed,90)),'batch_size':1,'precision':'FP32',
            'warmup':30,'repetitions':100,'scope':'model only; excludes IO and preprocessing',
            'samples_ms':elapsed},indent=2))
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    if not rows:raise RuntimeError('No finished training runs')
    (root/'metrics.json').write_text(json.dumps(rows,indent=2))
    with (root/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    with (root/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=predictions[0].keys());writer.writeheader();writer.writerows(predictions)
    plot_results(root,split,device,args.data)


def save(fig,path):
    fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)


def plot_results(root,split,device,data):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=root/'figures'
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for variant in COLORS:
        paths=sorted(root.glob(f'{variant}_seed*/history.csv'))
        histories=[np.genfromtxt(p,delimiter=',',names=True) for p in paths]
        for column,ax in zip(['train_loss','val_accuracy'],axes):
            values=np.stack([h[column] for h in histories])
            epochs=np.arange(1,values.shape[1]+1)
            ax.plot(epochs,values.mean(0),label=variant,color=COLORS[variant])
            if len(values)>1:ax.fill_between(epochs,values.min(0),values.max(0),alpha=.15,color=COLORS[variant])
            ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].set_ylabel('Training cross entropy');axes[1].set_ylabel('Validation accuracy')
    axes[1].legend();fig.tight_layout();save(fig,figures/'learning_curves.png')
    metrics=json.loads((root/'metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    variants=list(COLORS);conditions=['clean','dark','blur']
    for j,variant in enumerate(variants):
        means=[];std=[]
        for condition in conditions:
            vals=[r['accuracy']*100 for r in metrics if r['variant']==variant and r['condition']==condition]
            means.append(np.mean(vals));std.append(np.std(vals,ddof=1) if len(vals)>1 else 0)
        axes[0].bar(np.arange(3)+(j-1)*.24,means,.24,yerr=std,label=variant,color=COLORS[variant],capsize=3)
        record=next(r for r in metrics if r['variant']==variant)
        axes[1].scatter(record['parameters']/1e6,means[0],s=90,color=COLORS[variant],label=variant)
        axes[1].annotate(variant,(record['parameters']/1e6,means[0]),xytext=(5,5),textcoords='offset points')
    axes[0].set_xticks(range(3),['Clean','Brightness x0.6','Gaussian blur r=1'])
    axes[0].set_ylabel('Test accuracy (%)');axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Parameters (million)');axes[1].set_ylabel('Clean test accuracy (%)')
    fig.tight_layout();save(fig,figures/'comparison.png')
    # Display seed42 for all models; this selection is fixed before test evaluation.
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    short=['Annual','Forest','Herb.','Highway','Industr.','Pasture','Perm.','Resident.','River','SeaLake']
    for ax,variant in zip(axes,variants):
        run=root/f'{variant}_seed42'
        cm=np.load(run/'clean_confusion.npy');norm=cm/cm.sum(1,keepdims=True)
        ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,color='white' if norm[i,j]>.6 else 'black')
        ax.set_xticks(range(10),short,rotation=65,ha='right',fontsize=7)
        ax.set_yticks(range(10),short,fontsize=7);ax.set_title(variant);ax.set_xlabel('Predicted');ax.set_ylabel('True')
    fig.tight_layout();save(fig,figures/'confusion_matrices.png')
    fig,axes=plt.subplots(4,4,figsize=(10,10))
    samples=[]
    for label in ['AnnualCrop','Highway','Residential','River']:
        i=next(i for i in split['test'] if split['rows'][i]['label']==split['classes'].index(label))
        samples.append((i,label))
    evidence=[]
    for row,(idx,label) in enumerate(samples):
        path=Path(data)/split['rows'][idx]['path']
        image=np.array(Image.open(path).convert('RGB'))/255.
        axes[row,0].imshow(image);axes[row,0].set_title('True: '+label,fontsize=9)
        ds=Images(data,split,'test');tensor,_=ds[split['test'].index(idx)]
        evidence.append({'path':split['rows'][idx]['path'],'label':label})
        for col,variant in enumerate(variants,1):
            model,ckpt=load_model(root/f'{variant}_seed42/best.pt',device)
            cam,prob=gradcam(model,tensor[None].to(device))
            axes[row,col].imshow(image);axes[row,col].imshow(cam,cmap='jet',alpha=.45,vmin=0,vmax=1)
            axes[row,col].set_title(f'{variant}: {split["classes"][prob.argmax()]}\n{prob.max():.3f}',fontsize=8)
            del model
        for ax in axes[row]:ax.axis('off')
    fig.tight_layout();save(fig,figures/'gradcam_examples.png')
    (root/'gradcam_samples.json').write_text(json.dumps(evidence,indent=2))
    fig,axes=plt.subplots(2,5,figsize=(10,4.5))
    for label,ax in zip(split['classes'],axes.flat):
        row=next(r for r in split['rows'] if r['label']==split['classes'].index(label))
        ax.imshow(Image.open(Path(data)/row['path']));ax.set_title(label,fontsize=9);ax.axis('off')
    fig.tight_layout();save(fig,figures/'dataset_examples.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--results',default='results')
    evaluate(p.parse_args())
