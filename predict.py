"""Application: classify one RGB satellite tile and export a Grad-CAM figure."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from data import MEAN,STD
from analyze import load_model,gradcam

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--output',default='results/demo');a=p.parse_args()
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,ckpt=load_model(a.checkpoint,device)
    image=Image.open(a.image).convert('RGB').resize((64,64),Image.Resampling.BILINEAR)
    x=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])(image)[None].to(device)
    cam,prob=gradcam(model,x)
    order=np.argsort(prob)[::-1]
    result={'input':a.image,'checkpoint':a.checkpoint,'variant':ckpt['variant'],
            'predictions':[{'class':ckpt['classes'][i],'probability':float(prob[i])} for i in order],
            'scope':'single RGB satellite tile classification; not segmentation or object detection'}
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    (dest/'prediction.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(7,3))
    axes[0].imshow(image);axes[0].set_title('Input RGB tile')
    axes[1].imshow(image);axes[1].imshow(cam,cmap='jet',alpha=.45,vmin=0,vmax=1)
    axes[1].set_title(f'{ckpt["classes"][order[0]]}: {prob[order[0]]:.3f}')
    for ax in axes:ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'prediction.png',dpi=200);plt.close(fig)
    print(json.dumps(result,indent=2))
