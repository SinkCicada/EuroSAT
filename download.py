"""Atomic download, verified against published EuroSAT archive checksums."""
import hashlib,json,subprocess
from pathlib import Path

SOURCES=[
 ('https://hf-mirror.com/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip','c8fa014336c82ac7804f0398fcb19387'),
 ('https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip','c8fa014336c82ac7804f0398fcb19387'),
 ('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1','f46e308c4d50d4bf32fedad2d3d62f3b')]


def download(root=Path('data')):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    dest=root/'EuroSAT_RGB.zip'
    if dest.exists():
        actual=hashlib.md5(dest.read_bytes()).hexdigest()
        if actual in {digest for _,digest in SOURCES}:return
        raise RuntimeError('Existing archive failed checksum; preserve it and inspect before retrying.')
    temporary=root/'EuroSAT_RGB.zip.part'
    for url,digest in SOURCES:
        command=['curl','--fail','--location','--connect-timeout','15','--max-time','300','--retry','1','--output',str(temporary),url]
        print(json.dumps({'download_command':command}),flush=True)
        r=subprocess.run(command)
        if r.returncode==0 and hashlib.md5(temporary.read_bytes()).hexdigest()==digest:
            temporary.replace(dest)
            (root/'download_source.json').write_text(json.dumps({'url':url,'md5':digest,'bytes':dest.stat().st_size,
                'canonical_dataset':'https://zenodo.org/records/7711810',
                'torchvision_source':'https://github.com/pytorch/vision/blob/v0.22.1/torchvision/datasets/eurosat.py'},indent=2))
            print('DOWNLOAD_VERIFIED',flush=True);return
    raise RuntimeError('All sources failed; inspect partial download and network access.')


if __name__=='__main__':download()
