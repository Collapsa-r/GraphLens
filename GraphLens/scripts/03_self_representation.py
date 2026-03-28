#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.self_representation import LinearSRConfig, linear_self_representation

def _ensure_dir(p:str): os.makedirs(p,exist_ok=True)

def _load_embeddings(path:str)->np.ndarray:
    obj=torch.load(path,map_location='cpu')
    if torch.is_tensor(obj): X=obj.detach().cpu().numpy()
    elif isinstance(obj,dict):
        for k in ['X','emb','embeddings','features','H']:
            if k in obj: X=np.asarray(obj[k]); break
        else: raise ValueError("No 'X'/'emb'/'embeddings'/'features'/'H' found in input file")
    else: X=np.asarray(obj)
    if X.ndim!=2: raise ValueError(f'Expected a 2D embedding array, got shape={X.shape}')
    return X.astype(np.float32,copy=False)

def _save_repr(A,Z,out_dir:str,tag:str,meta:dict)->str:
    _ensure_dir(out_dir); out_pt=os.path.join(out_dir,f'representation_matrix_{tag}.pt'); torch.save({'A':A,'Z':Z,'meta':meta},out_pt); info_json=os.path.join(out_dir,f'representation_matrix_{tag}_info.json')
    with open(info_json,'w',encoding='utf-8') as f: json.dump(meta,f,indent=2)
    print(f'[OK] representation -> {out_pt}'); print(f'[OK] info -> {info_json}'); return out_pt

def build_argparser():
    ap=argparse.ArgumentParser('03: linear self-representation')
    ap.add_argument('--emb',type=str,required=True); ap.add_argument('--tag',type=str,default=None); ap.add_argument('--out_dir',type=str,default='./outputs/representation_matrices')
    ap.add_argument('--k',type=int,default=2); ap.add_argument('--alpha',type=float,default=0.0); ap.add_argument('--beta',type=float,default=3.0); ap.add_argument('--gamma',type=float,default=0.0); ap.add_argument('--lambda1',type=float,default=0.0)
    ap.add_argument('--iters',type=int,default=60); ap.add_argument('--tol',type=float,default=2e-4); ap.add_argument('--enforce_z_nonneg',action='store_true'); ap.add_argument('--enforce_z_colsum1',action='store_true'); ap.add_argument('--topk_sparsify',type=int,default=0)
    ap.add_argument('--init_topk',type=int,default=24); ap.add_argument('--init_mutual_knn',action='store_true')
    ap.add_argument('--symmetrize',type=str,choices=['avg','max','min'],default='avg'); ap.add_argument('--normalize',type=str,choices=['none','row','col'],default='none')
    return ap

def main():
    args=build_argparser().parse_args(); X=_load_embeddings(args.emb); tag=args.tag or Path(args.emb).stem.replace('emb_','').replace('embeddings_','')
    cfg=LinearSRConfig(k=args.k,alpha=args.alpha,beta=args.beta,gamma=args.gamma,lambda1=args.lambda1,iters=args.iters,tol=args.tol,enforce_z_nonneg=bool(args.enforce_z_nonneg),enforce_z_colsum1=bool(args.enforce_z_colsum1),topk_sparsify=args.topk_sparsify,init_topk=args.init_topk,init_mutual_knn=bool(args.init_mutual_knn),symmetrize=args.symmetrize,normalize=args.normalize,verbose=True)
    A,Z,meta=linear_self_representation(X,cfg); _save_repr(A,Z,args.out_dir,tag,meta)
if __name__=='__main__': main()
