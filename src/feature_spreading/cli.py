import argparse

from . import DATASETS, MODEL

def main():
    parser = argparse.ArgumentParser(prog='fs')
    sub    = parser.add_subparsers(dest='cmd', required=True)
    
    p = sub.add_parser('download', help='download dataset from huggingface + extract CLIP features')
    p.add_argument('dataset',       type=str, choices=DATASETS)
    p.add_argument('--model',       type=str, default=MODEL)
    p.add_argument('--n_obs',       type=int, default=10_000)
    p.add_argument('--seed',        type=int, default=123)
    
    p = sub.add_parser('run', help='run baseline + feature spreading')
    p.add_argument('dataset',       type=str, choices=DATASETS)
    p.add_argument('--model',       type=str, default=MODEL)
    p.add_argument('--n_obs',       type=int, default=5_000)
    p.add_argument('--n_per_class', type=int, default=1)
    p.add_argument('--seed',        type=int, default=234)
    
    p = sub.add_parser('agop', help='run feature spreading w/ AGOP-learned graph metric')
    p.add_argument('dataset',       type=str, choices=DATASETS)
    p.add_argument('--model',       type=str, default=MODEL)
    p.add_argument('--n_obs',       type=int, default=5_000)
    p.add_argument('--n_per_class', type=int, default=1)
    p.add_argument('--seed',        type=int, default=234)
    p.add_argument('--agop',        type=str,   default='kernel', choices=['kernel', 'linear'])
    p.add_argument('--n_iters',     type=int,   default=4)
    p.add_argument('--lam',         type=float, default=0.5)  # 0 = cosine metric, 1 = pure AGOP metric
    p.add_argument('--krr_reg',     type=float, default=1e-3)
    
    args = vars(parser.parse_args())
    cmd  = args.pop('cmd')
    
    if cmd == 'download':
        from .download import download
        download(**args)
    elif cmd == 'run':
        from .run import run
        run(**args)
    elif cmd == 'agop':
        from .agop import run_agop
        run_agop(**args)
