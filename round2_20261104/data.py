"""Official RGB archive, exact-file deduplication, fixed stratified split."""
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
URL = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
MD5 = 'f46e308c4d50d4bf32fedad2d3d62f3b'
TORCHVISION_MD5 = 'c8fa014336c82ac7804f0398fcb19387'


def prepare(root='data'):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    archive = root / 'EuroSAT_RGB.zip'
    if not archive.exists():
        from download import download
        download(root)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest not in (MD5, TORCHVISION_MD5):
        raise RuntimeError(f'Archive checksum mismatch: {digest}')
    extracted = root / 'raw'
    if not (extracted / '.complete').exists():
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (extracted / member.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise RuntimeError('Archive path outside data directory')
            z.extractall(extracted)
        (extracted / '.complete').write_text(digest)
    files = sorted(p for p in extracted.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg') and not p.name.startswith('.'))
    classes = sorted({p.parent.name for p in files})
    if len(classes) != 10 or len(files) != 27000:
        raise RuntimeError(f'Unexpected dataset: classes={len(classes)}, images={len(files)}')
    manifest = root / 'split.json'
    if manifest.exists():
        return json.loads(manifest.read_text())
    seen, rows, duplicates = {}, [], []
    for p in files:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha in seen:
            duplicates.append({'excluded': str(p.relative_to(root)), 'kept': seen[sha]})
            continue
        rel = str(p.relative_to(root))
        seen[sha] = rel
        rows.append({'path': rel, 'label': classes.index(p.parent.name), 'sha256': sha})
    indices = list(range(len(rows)))
    train, other = train_test_split(indices, test_size=0.3, random_state=2026,
                                   stratify=[r['label'] for r in rows])
    val, test = train_test_split(other, test_size=0.5, random_state=2026,
                                stratify=[rows[i]['label'] for i in other])
    download_meta = root / 'download_source.json'
    source = json.loads(download_meta.read_text()) if download_meta.exists() else {'url': URL}
    split = {'source_url': source['url'], 'archive_md5': digest, 'archive_bytes': archive.stat().st_size,
             'classes': classes, 'raw_count': len(files), 'duplicates_removed': duplicates,
             'rows': rows, 'train': train, 'val': val, 'test': test, 'split_seed': 2026,
             'counts': {k: dict(Counter(classes[rows[i]['label']] for i in ids))
                        for k, ids in [('train', train), ('val', val), ('test', test)]}}
    manifest.write_text(json.dumps(split, indent=2))
    return split


class Images(Dataset):
    def __init__(self, root, split, mode, corruption='clean'):
        self.root, self.split, self.mode = Path(root), split, mode
        self.indices = split[mode]
        self.corruption = corruption
        # Rotation is sampled discretely below, avoiding interpolation artifacts.
        self.convert = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        row = self.split['rows'][self.indices[j]]
        with Image.open(self.root / row['path']) as image:
            image = image.convert('RGB')
            if self.mode == 'train':
                if random.random() < 0.5: image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if random.random() < 0.5: image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image = image.rotate(random.choice([0, 90, 180, 270]))
            if self.corruption == 'dark': image = ImageEnhance.Brightness(image).enhance(0.6)
            elif self.corruption == 'blur': image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
            elif self.corruption != 'clean': raise ValueError(self.corruption)
            return self.convert(image), row['label']


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
