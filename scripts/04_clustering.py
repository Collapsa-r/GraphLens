#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, random, sys
from pathlib import Path
import numpy as np
import torch
from scipy import sparse as sp
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.clustering import affinity_from_Z, spectral_cluster

def set_seed(seed:int=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _is_sparse(x): return sp.issparse(x) or bool(torch.is_tensor(x) and x.is_sparse)

def _to_dense_numpy(x):
    if sp.issparse(x): return x.toarray()
    if torch.is_tensor(x): return x.to_dense().cpu().numpy() if x.is_sparse else x.cpu().numpy()
    return np.asarray(x)

def load_matrix(rep_path:Path, source_key:str)->np.ndarray:
    src=source_key.upper(); d=torch.load(str(rep_path),map_location='cpu'); M=None
    if isinstance(d,dict):
        prefer=[src]
        if src=='A': prefer += ['AFFINITY','A_norm','A_sym','A0']
        elif src=='Z': prefer += ['Z','REPRESENTATION','C','COEF']
        for k in prefer:
            if k in d: M=d[k]; break
        if M is None:
            for _,v in d.items():
                if hasattr(v,'shape') or _is_sparse(v) or torch.is_tensor(v): M=v; break
        if M is None: raise KeyError(f'cannot find matrix for source={src} in {rep_path}')
    else: M=d
    return _to_dense_numpy(M).astype(np.float64,copy=False)

def main():
    parser=argparse.ArgumentParser('04: spectral clustering')
    parser.add_argument('--rep',type=str,required=True); parser.add_argument('--k',type=int,required=True); parser.add_argument('--source',type=str,default='A',choices=['A','Z']); parser.add_argument('--normalize',type=str,default='none',choices=['none','row','col']); parser.add_argument('--symmetrize',type=str,default='avg',choices=['avg','max','min']); parser.add_argument('--kmeans-n-init',type=int,default=300); parser.add_argument('--seed',type=int,default=0); parser.add_argument('--tag',type=str,required=True); parser.add_argument('--out_dir',type=str,default='./outputs/clusters'); parser.add_argument('--save_affinity',action='store_true')
    args=parser.parse_args(); set_seed(args.seed); rep_path=Path(args.rep); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    M=load_matrix(rep_path,args.source)
    A=M.copy() if args.source.upper()=='A' else affinity_from_Z(M,normalize=args.normalize,symmetrize=args.symmetrize,self_loop=1.0)
    labels,U=spectral_cluster(A,k=args.k,n_init=args.kmeans_n_init,seed=args.seed,return_U=True)
    lab_path=out_dir/f'labels_{args.tag}_{args.source.upper()}_seed{args.seed}.pt'; torch.save({'labels': torch.as_tensor(labels,dtype=torch.long)},str(lab_path)); print(f'[OK] labels -> {lab_path}')
    if args.save_affinity:
        aff_path=out_dir/f'affinity_{args.tag}_{args.source.upper()}_seed{args.seed}.pt'; torch.save(A,str(aff_path)); print(f'[OK] affinity -> {aff_path}')
    emb_path=out_dir/f'spectral_embedding_{args.tag}_{args.source.upper()}_seed{args.seed}.pt'; torch.save({'U': torch.as_tensor(U,dtype=torch.float32)},str(emb_path)); print(f'[OK] spectral embedding -> {emb_path}')
if __name__=='__main__': main()
