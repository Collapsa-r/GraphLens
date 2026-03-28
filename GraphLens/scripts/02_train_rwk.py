#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.graph_kernel_network import GKNConfig, train_model

def load_manifest(manifest_pt: Optional[str]) -> Optional[Dict[str, Any]]:
    if not manifest_pt: return None
    obj=torch.load(manifest_pt,map_location='cpu')
    if not isinstance(obj,dict): raise ValueError('manifest .pt must be a dict')
    return obj

def load_subgraphs(path:str)->List[Dict[str,Any]]:
    obj=torch.load(path,map_location='cpu')
    if isinstance(obj,list): items=obj
    elif isinstance(obj,dict) and 'items' in obj: items=obj['items']
    else: raise ValueError("subgraphs_*.pt must be a list or a dict containing key 'items'")
    for it in items:
        if isinstance(it.get('edge_index',None),torch.Tensor): it['edge_index']=it['edge_index'].long()
        if isinstance(it.get('x',None),torch.Tensor):
            x=it['x'].float()
            if x.dim()==1: x=x.view(-1,1)
            it['x']=x
    return items

def filter_by_manifest(items, manifest):
    if manifest is None: return items
    allow=None
    if 'graph_ids' in manifest: allow=set(int(x) for x in torch.as_tensor(manifest['graph_ids']).view(-1).tolist())
    elif 'indices' in manifest: allow=set(int(x) for x in torch.as_tensor(manifest['indices']).view(-1).tolist())
    if allow is None: return items
    if len(items)>0 and 'graph_id' in items[0]: return [it for it in items if int(it.get('graph_id',-1)) in allow]
    return items

def build_argparser():
    p=argparse.ArgumentParser('02: train graph kernel network')
    p.add_argument('--subgraphs',required=True)
    p.add_argument('--manifest',default=None)
    p.add_argument('--out_dir',default='./outputs')
    p.add_argument('--tag',default=None)
    p.add_argument('--export-emb-path',type=str,default=None)
    p.add_argument('--layers',type=int,default=2); p.add_argument('--num-filters',type=int,default=8); p.add_argument('--walk-len',type=int,default=3); p.add_argument('--gamma',type=float,default=0.7); p.add_argument('--embed-dim',type=int,default=128); p.add_argument('--dropout',type=float,default=0.10); p.add_argument('--use-layernorm',action='store_true'); p.add_argument('--no-base',action='store_true'); p.add_argument('--readout',choices=['concat','last','sum'],default='concat'); p.add_argument('--gate-tau',type=float,default=1.0); p.add_argument('--residual-cat',action='store_true')
    p.add_argument('--max-motifs-per-type',type=int,default=-1); p.add_argument('--motif-sample-ratio',type=float,default=1.0)
    p.add_argument('--recon-graph',choices=['mse','bce','off'],default='mse'); p.add_argument('--recon-motif',choices=['mse','bce','off'],default='off'); p.add_argument('--lambda-motif',type=float,default=1.0); p.add_argument('--recon-x',action='store_true'); p.add_argument('--lambda-x',type=float,default=1e-3); p.add_argument('--lambda-div',type=float,default=1e-3)
    p.add_argument('--epochs',type=int,default=80); p.add_argument('--lr',type=float,default=2e-3); p.add_argument('--weight-decay',type=float,default=5e-4); p.add_argument('--scheduler',choices=['none','plateau','cosine'],default='plateau'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--early-patience',type=int,default=20); p.add_argument('--early-delta',type=float,default=5e-4); p.add_argument('--seed',type=int,default=0)
    return p

def main():
    args=build_argparser().parse_args(); os.makedirs(args.out_dir,exist_ok=True); dir_emb=os.path.join(args.out_dir,'embeddings'); os.makedirs(dir_emb,exist_ok=True)
    tag=args.tag or Path(args.subgraphs).stem.replace('subgraphs_',''); emb_path=args.export_emb_path if args.export_emb_path else os.path.join(dir_emb,f'emb_{tag}.pt')
    items=load_subgraphs(args.subgraphs); manifest=load_manifest(args.manifest) if args.manifest else None; items=filter_by_manifest(items,manifest)
    cfg=GKNConfig(n_layers=args.layers,n_filters=args.num_filters,walk_len=args.walk_len,gamma=args.gamma,embed_dim=args.embed_dim,drop_prob=args.dropout,use_layernorm=bool(args.use_layernorm),use_base=not bool(args.no_base),readout=args.readout,gate_tau=args.gate_tau,residual_cat=bool(args.residual_cat),max_motifs_per_type=args.max_motifs_per_type,motif_sample_ratio=args.motif_sample_ratio,recon_graph=args.recon_graph,recon_motif=args.recon_motif,lambda_motif=args.lambda_motif,recon_x=bool(args.recon_x),lambda_x=args.lambda_x,lambda_div=args.lambda_div,epochs=args.epochs,lr=args.lr,weight_decay=args.weight_decay,scheduler=args.scheduler,early_patience=args.early_patience,early_delta=args.early_delta,seed=args.seed,verbose=True,batch_log_every=50)
    _,X=train_model(items,cfg=cfg,device=args.device); torch.save({'X':X},emb_path); print(f'[OK] embeddings -> {emb_path} shape={tuple(X.shape)}')
    info_path=emb_path.replace('.pt','_info.json')
    with open(info_path,'w',encoding='utf-8') as f: json.dump({'tag':tag,'subgraphs_path':args.subgraphs,'manifest_path':args.manifest,'emb_path':emb_path,'num_items':int(X.size(0)),'embed_dim':int(X.size(1)),'cfg':vars(args)},f,ensure_ascii=False,indent=2)
    print(f'[OK] info -> {info_path}')
if __name__=='__main__': main()
