import json
from pathlib import Path
import torch
from data import prepare, Images
from models import make_model

torch.set_num_threads(4)
split=prepare()
assert not (set(split['train']) & set(split['val']) or set(split['train']) & set(split['test']) or set(split['val']) & set(split['test']))
assert sum(len(split[k]) for k in ['train','val','test'])==len(split['rows'])
hash_sets=[{split['rows'][i]['sha256'] for i in split[k]} for k in ['train','val','test']]
assert all(not hash_sets[i]&hash_sets[j] for i in range(3) for j in range(i))
x,y=Images('data',split,'train')[0]
assert x.shape==(3,64,64)
print(json.dumps({'raw_images':split['raw_count'],'unique_images':len(split['rows']),
                 'split_sizes':{k:len(split[k]) for k in ['train','val','test']},'classes':split['classes']}),flush=True)
device='cuda' if torch.cuda.is_available() else 'cpu'
for variant in ['baseline','se','cbam']:
    torch.manual_seed(42)
    model=make_model(variant).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    output=model(x.unsqueeze(0).repeat(2,1,1,1).to(device))
    loss=torch.nn.functional.cross_entropy(output,torch.tensor([y,y],device=device))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    path=Path('results/smoke')/f'{variant}.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),path)
    restored=make_model(variant).to(device)
    restored.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    model.eval();restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x[None].to(device)),restored(x[None].to(device)))
    print(json.dumps({'variant':variant,'loss':loss.item(),'output_shape':list(output.shape),
                     'backward_finite':True,'checkpoint_reload_matches':True}),flush=True)
    del model,restored,optimizer
    if device=='cuda':torch.cuda.empty_cache()
