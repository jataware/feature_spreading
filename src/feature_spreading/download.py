"""
  download.py - pull images from huggingface, featurize w/ CLIP, save X.npy / y.npy
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

from . import DATA_DIR, DATASETS

def download(dataset, model, n_obs, seed, batch_size=64):
    outdir = os.path.join(DATA_DIR, dataset, model)
    os.makedirs(outdir, exist_ok=True)
    
    ds = load_dataset(DATASETS[dataset], split='train')
    ds = ds.shuffle(seed=seed).select(range(min(n_obs, len(ds))))
    
    device    = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    clip      = CLIPModel.from_pretrained(model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model)
    
    X = []
    for i in tqdm(range(0, len(ds), batch_size), desc=dataset):
        imgs = [img.convert('RGB') for img in ds[i:i + batch_size]['image']]
        with torch.no_grad():
            inputs = processor(images=imgs, return_tensors='pt').to(device)
            feats  = clip.get_image_features(**inputs)
            if not torch.is_tensor(feats): # transformers>=5 returns a ModelOutput
                feats = feats.pooler_output
        
        X.append(feats.float().cpu().numpy())
    
    X = np.concatenate(X)
    y = np.array(ds['label'])
    
    np.save(os.path.join(outdir, 'X.npy'), X)
    np.save(os.path.join(outdir, 'y.npy'), y)
    print(f'[download] {dataset} -> {outdir} | X={X.shape} y={y.shape} n_classes={len(np.unique(y))}')
