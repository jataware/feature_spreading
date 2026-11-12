"""
  data.py - dataset registry + "get me vectors" for the blog experiments

  usage (from repo root): python blog/data.py <name> [--encoder <hf model id>]

  Everything ends up as data/<name>/<encoder>/{X,y}.npy, at most 10,000 rows (random subset, fixed seed).
  This is the only modality-specific code in the project: the experiments only ever see (X, y).
  Skips work that is already on disk.
"""

import io
import os
import sys
import argparse
import numpy as np

DATA_DIR = 'data'
MAX_ROWS = 10_000
SEED     = 123

CLIP, DINO   = 'openai/clip-vit-large-patch14', 'facebook/dinov2-base'
QWEN, MINILM = 'Qwen/Qwen3-Embedding-0.6B', 'sentence-transformers/all-MiniLM-L6-v2'
CLAP         = 'laion/clap-htsat-unfused'
GEMINI       = 'google/gemini-embedding-001'    # not local: called through OpenRouter, needs OPENROUTER_API_KEY in .secrets

IMAGES = {
    'pets'          : dict(kind='image', hf='Multimodal-Fatima/OxfordPets_train',   split='train', x='image', y='label'),
    'cars'          : dict(kind='image', hf='Multimodal-Fatima/StanfordCars_train', split='train', x='image', y='label'),
    'resisc45'      : dict(kind='image', hf='jonathan-roberts1/NWPU-RESISC45',      split='train', x='image', y='label'),
    'flowers'       : dict(kind='image', hf='nkirschi/oxford-flowers',              split='train', x='image', y='label'),
    'food101'       : dict(kind='image', hf='ethz/food101',                         split='train', x='image', y='label'),
    'fashion_mnist' : dict(kind='image', hf='zalando-datasets/fashion_mnist',       split='train', x='image', y='label'),
}
TEXTS = {
    'newsgroups'    : dict(kind='text', hf='SetFit/20_newsgroups', split='train', x='text',    y='label'),
    'banking77'     : dict(kind='text', hf='PolyAI/banking77',     split='train', x='text',    y='label'),
    'clinc150'      : dict(kind='text', hf='clinc/clinc_oos',      split='train', x='text',    y='intent', config='plus', drop_label='oos'),
    'dbpedia14'     : dict(kind='text', hf='fancyzhx/dbpedia_14',  split='test',  x='content', y='label'),
    'agnews'        : dict(kind='text', hf='fancyzhx/ag_news',     split='train', x='text',    y='label'),
    'emotion'       : dict(kind='text', hf='dair-ai/emotion',      split='train', x='text',    y='label',  config='split'),
}

# name -> where the raw data comes from, which encoder turns it into vectors, and which group it is in.
#   dev   : every default and every grid setting is chosen on these (CLIP image embeddings)
#   other : everything else; settings are never chosen on these
# `store` is the folder under data/ (several entries can share raw data and differ only in the encoder).
REGISTRY = {
    **{name               : dict(group='dev',   encoder=CLIP,   store=name, **cfg) for name, cfg in IMAGES.items()},
    **{name + '_dinov2'   : dict(group='other', encoder=DINO,   store=name, **cfg) for name, cfg in IMAGES.items()},   # same images, different encoder
    **{name               : dict(group='other', encoder=QWEN,   store=name, **cfg) for name, cfg in TEXTS.items()},
    **{name + '_minilm'   : dict(group='other', encoder=MINILM, store=name, **cfg) for name, cfg in TEXTS.items()},    # same text, much weaker encoder
    **{name + '_gemini'   : dict(group='other', encoder=GEMINI, store=name, **cfg) for name, cfg in TEXTS.items()},    # same text, hosted encoder
    'esc50'  : dict(group='other', kind='audio',  encoder=CLAP,  store='esc50',  hf='ashraq/esc50', split='train', x='audio', y='target'),
    'mnist'  : dict(group='other', kind='openml', encoder='raw', store='mnist',  openml=('mnist_784', 1), scale=255.0),
    'isolet' : dict(group='other', kind='openml', encoder='raw', store='isolet', openml=('isolet', 1)),
    'har'    : dict(group='other', kind='openml', encoder='raw', store='har',    openml=('har', 1)),
}

def path(name, encoder=None):
    return os.path.join(DATA_DIR, REGISTRY[name]['store'], encoder or REGISTRY[name]['encoder'])

def exists(name, encoder=None):
    return os.path.exists(os.path.join(path(name, encoder), 'X.npy'))

def load(name, encoder=None):
    p = path(name, encoder)
    return np.load(os.path.join(p, 'X.npy')), np.load(os.path.join(p, 'y.npy'))

# --
# Featurizers

def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def _batches(n, bs):
    from tqdm import tqdm
    return tqdm(range(0, n, bs))

def embed_text(texts, encoder, bs=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(encoder), AutoModel.from_pretrained(encoder).to(_device()).eval()
    last_token = 'Qwen' in encoder
    if last_token:
        tok.padding_side = 'left'          # so the last position is always a real token
    out = []
    for i in _batches(len(texts), bs or (16 if last_token else 128)):
        enc = tok(texts[i:i + (bs or (16 if last_token else 128))], padding=True, truncation=True, max_length=256, return_tensors='pt').to(_device())
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        if last_token:            # Qwen3-Embedding: last token
            e = h[:,-1]
        elif 'bge' in encoder:    # bge: [CLS] token
            e = h[:,0]
        else:                     # sentence-transformers MiniLM: mean over real tokens
            m = enc['attention_mask'][:,:,None].float()
            e = (h * m).sum(1) / m.sum(1)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)

def embed_text_api(texts, encoder, cache_dir, bs=64, max_chars=1200):
    """ hosted embedding model through OpenRouter. Sends the first max_chars characters of each text (about the 256 tokens
        the local encoders see). Finished batches are cached on disk, so a failed run resumes without paying twice. """
    import json
    import time
    import urllib.request
    key = [l.split('=', 1)[1].strip().strip('"\'') for l in open('.secrets') if 'OPENROUTER_API_KEY' in l][0]
    os.makedirs(cache_dir, exist_ok=True)
    out, tokens = [], 0
    for i in _batches(len(texts), bs):
        f = os.path.join(cache_dir, f'{i:07d}.npy')
        if not os.path.exists(f):
            batch = [(t[:max_chars] if t.strip() else '(empty)') for t in texts[i:i + bs]]
            req   = urllib.request.Request('https://openrouter.ai/api/v1/embeddings', method='POST',
                                           data=json.dumps({'model' : encoder, 'input' : batch}).encode(),
                                           headers={'Authorization' : f'Bearer {key}', 'Content-Type' : 'application/json'})
            j, err = None, None
            for attempt in range(6):
                try:
                    j = json.loads(urllib.request.urlopen(req, timeout=120).read())
                    if 'data' in j:
                        break
                    err = str(j)[:200]
                except Exception as e:
                    err = f'{type(e).__name__}: {e}'
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f'embedding request failed: {err}')
            tokens += j.get('usage', {}).get('total_tokens', 0)
            np.save(f, np.array([e['embedding'] for e in sorted(j['data'], key=lambda e: e['index'])], dtype=np.float32))
        out.append(np.load(f))
    print(f'[data] {encoder}: {tokens:,} tokens billed this run')
    return np.concatenate(out)

def embed_image(images, encoder, bs=64):
    import torch
    from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
    is_clip = 'clip' in encoder
    proc    = (CLIPProcessor if is_clip else AutoImageProcessor).from_pretrained(encoder)
    model   = (CLIPModel if is_clip else AutoModel).from_pretrained(encoder).to(_device()).eval()
    out = []
    for i in _batches(len(images), bs):
        imgs = [im.convert('RGB') for im in images[i:i + bs]]
        with torch.no_grad():
            enc = proc(images=imgs, return_tensors='pt').to(_device())
            if is_clip:
                e = model.get_image_features(**enc)
                e = e if torch.is_tensor(e) else e.pooler_output
            else:                 # DINOv2: [CLS] token
                e = model(**enc).last_hidden_state[:,0]
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)

def embed_audio(clips, encoder, bs=16):
    """ clips: list of {'bytes': ...} (undecoded audio files) """
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    from transformers import ClapModel, ClapProcessor
    proc, model = ClapProcessor.from_pretrained(encoder), ClapModel.from_pretrained(encoder).to(_device()).eval()
    out = []
    for i in _batches(len(clips), bs):
        wavs = []
        for c in clips[i:i + bs]:
            w, sr = sf.read(io.BytesIO(c['bytes']), dtype='float32', always_2d=True)
            w     = w.mean(1)
            if sr != 48_000:
                g = gcd(48_000, sr)
                w = resample_poly(w, 48_000 // g, sr // g).astype(np.float32)
            wavs.append(w)
        with torch.no_grad():
            enc = proc(audio=wavs, sampling_rate=48_000, return_tensors='pt').to(_device())
            e   = model.get_audio_features(**enc)
            e   = e if torch.is_tensor(e) else e.pooler_output
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)

# --
# Fetch + featurize one dataset

def build(name, encoder=None):
    cfg     = REGISTRY[name]
    encoder = encoder or cfg['encoder']
    outdir  = path(name, encoder)
    if exists(name, encoder):
        print(f'[data] {name} / {encoder}: already on disk, skipping')
        return

    if cfg['kind'] == 'openml':
        from sklearn.datasets import fetch_openml
        d    = fetch_openml(cfg['openml'][0], version=cfg['openml'][1], as_frame=False, parser='liac-arff')   # the pandas parser trips on integer categories with pandas 3
        X    = d.data.astype(np.float32) / cfg.get('scale', 1.0)
        _, y = np.unique(d.target, return_inverse=True)
        p    = np.random.default_rng(SEED).permutation(len(X))[:MAX_ROWS]
        X, y = X[p], y[p]
    else:
        from datasets import load_dataset, Audio
        ds = load_dataset(cfg['hf'], cfg.get('config'), split=cfg['split'])
        if 'drop_label' in cfg:
            drop = ds.features[cfg['y']].names.index(cfg['drop_label'])
            ds   = ds.filter(lambda r: r[cfg['y']] != drop)
        ds = ds.shuffle(seed=SEED).select(range(min(MAX_ROWS, len(ds))))
        if cfg['kind'] == 'audio':
            ds = ds.cast_column(cfg['x'], Audio(decode=False))

        _, y = np.unique(np.array(ds[cfg['y']]), return_inverse=True)
        items = ds[cfg['x']]
        if cfg['kind'] == 'text' and encoder == GEMINI:
            X = embed_text_api(list(items), encoder, cache_dir=os.path.join(outdir, '_parts'))
        else:
            X = dict(text=embed_text, image=embed_image, audio=embed_audio)[cfg['kind']](list(items) if cfg['kind'] != 'image' else items, encoder)

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'X.npy'), X)
    np.save(os.path.join(outdir, 'y.npy'), y)
    print(f'[data] {name} / {encoder} -> {outdir} | X={X.shape} y={y.shape} n_classes={len(np.unique(y))}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+', choices=list(REGISTRY))
    parser.add_argument('--encoder', type=str, default=None)
    args = parser.parse_args()
    for name in args.names:
        build(name, args.encoder)
