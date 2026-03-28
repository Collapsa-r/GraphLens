#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.subgraph_extraction import ExtractConfig, extract_dataset_to_cache
from models.graph_kernel_network import GKNConfig, train_model
from models.self_representation import LinearSRConfig, linear_self_representation
from models.clustering import affinity_from_Z, spectral_cluster

def _ensure_dir(path:str): Path(path).mkdir(parents=True,exist_ok=True)

def build_argparser():
    ap=argparse.ArgumentParser('end-to-end pipeline')
    ap.add_argument('--dataset',type=str,required=True); ap.add_argument('--k',type=int,required=True); ap.add_argument('--tag',type=str,required=True); ap.add_argument('--data_root',type=str,default='./data/TUDataset'); ap.add_argument('--out_root',type=str,default='./outputs'); ap.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--manifest_path',type=str,default='')
    ap.add_argument('--skip_step01',action='store_true'); ap.add_argument('--skip_step02',action='store_true'); ap.add_argument('--skip_step03',action='store_true'); ap.add_argument('--skip_step04',action='store_true'); ap.add_argument('--subgraphs_cache',type=str,default=''); ap.add_argument('--embeddings_cache',type=str,default=''); ap.add_argument('--representation_cache',type=str,default='')
    ap.add_argument('--motifs',nargs='+',default=['triangle','wedge','cycle4']); ap.add_argument('--z_threshold',type=float,default=1.0); ap.add_argument('--null_iters',type=int,default=20); ap.add_argument('--edge_swaps_per_edge',type=float,default=5.0); ap.add_argument('--pairwise_max_dist',type=int,default=2); ap.add_argument('--min_keep_per_type',type=int,default=0)
    ap.add_argument('--layers',type=int,default=2); ap.add_argument('--num_filters',type=int,default=8); ap.add_argument('--walk_len',type=int,default=3); ap.add_argument('--gamma',type=float,default=0.7); ap.add_argument('--embed_dim',type=int,default=128); ap.add_argument('--dropout',type=float,default=0.10); ap.add_argument('--use_layernorm',action='store_true'); ap.add_argument('--no_base',action='store_true'); ap.add_argument('--readout',type=str,default='concat',choices=['concat','last','sum']); ap.add_argument('--gate_tau',type=float,default=1.0); ap.add_argument('--residual_cat',action='store_true'); ap.add_argument('--max_motifs_per_type',type=int,default=-1); ap.add_argument('--motif_sample_ratio',type=float,default=1.0); ap.add_argument('--recon_graph',type=str,default='mse',choices=['mse','bce','off']); ap.add_argument('--recon_motif',type=str,default='off',choices=['mse','bce','off']); ap.add_argument('--lambda_motif',type=float,default=1.0); ap.add_argument('--recon_x',action='store_true'); ap.add_argument('--lambda_x',type=float,default=1e-3); ap.add_argument('--lambda_div',type=float,default=1e-3); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--weight_decay',type=float,default=5e-4); ap.add_argument('--scheduler',type=str,default='plateau',choices=['none','plateau','cosine']); ap.add_argument('--early_patience',type=int,default=20); ap.add_argument('--early_delta',type=float,default=5e-4); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--sr_alpha',type=float,default=0.0); ap.add_argument('--sr_beta',type=float,default=3.0); ap.add_argument('--sr_gamma',type=float,default=0.0); ap.add_argument('--sr_lambda1',type=float,default=0.0); ap.add_argument('--sr_iters',type=int,default=60); ap.add_argument('--sr_tol',type=float,default=2e-4); ap.add_argument('--enforce_z_nonneg',action='store_true'); ap.add_argument('--enforce_z_colsum1',action='store_true'); ap.add_argument('--topk_sparsify',type=int,default=0); ap.add_argument('--init_topk',type=int,default=24); ap.add_argument('--init_mutual_knn',action='store_true'); ap.add_argument('--symmetrize',type=str,default='avg',choices=['avg','max','min']); ap.add_argument('--normalize',type=str,default='none',choices=['none','row','col'])
    ap.add_argument('--cluster_source',type=str,default='Z',choices=['A','Z']); ap.add_argument('--cluster_seed',type=int,default=0); ap.add_argument('--kmeans_n_init',type=int,default=300)
    return ap

def main():
    args=build_argparser().parse_args(); _ensure_dir(args.out_root); cache_dir=Path(args.out_root)/'cache'; emb_dir=Path(args.out_root)/'embeddings'; rep_dir=Path(args.out_root)/'representation_matrices'; clu_dir=Path(args.out_root)/'clusters'
    for d in [cache_dir,emb_dir,rep_dir,clu_dir]: d.mkdir(parents=True,exist_ok=True)
    if args.skip_step01:
        subgraphs_pt=args.subgraphs_cache
        if not subgraphs_pt or not os.path.isfile(subgraphs_pt): raise FileNotFoundError(f'cached subgraphs not found: {subgraphs_pt}')
        print(f'[Step-01] skip -> {subgraphs_pt}')
    else:
        print('\n[Step-01] motif extraction'); cfg1=ExtractConfig(data_root=args.data_root,out_root=args.out_root,motifs=tuple(args.motifs),z_threshold=args.z_threshold,null_iters=args.null_iters,edge_swaps_per_edge=args.edge_swaps_per_edge,pairwise_max_dist=args.pairwise_max_dist,min_keep_per_type=args.min_keep_per_type)
        ret1=extract_dataset_to_cache(dataset_name=args.dataset,tag=args.tag,cfg=cfg1,sample_manifest=args.manifest_path if args.manifest_path else None); subgraphs_pt=ret1['cache_path']; args.manifest_path=ret1['manifest_path']; print(f'[Step-01] cache -> {subgraphs_pt}')
    emb_pt=str(emb_dir/f'emb_{args.tag}.pt')
    if args.skip_step02:
        if args.embeddings_cache and os.path.isfile(args.embeddings_cache): emb_pt=args.embeddings_cache
        else: raise FileNotFoundError(f'cached embeddings not found: {args.embeddings_cache}')
        print(f'[Step-02] skip -> {emb_pt}')
    else:
        print('\n[Step-02] graph kernel network'); obj=torch.load(subgraphs_pt,map_location='cpu'); items=obj['items'] if isinstance(obj,dict) and 'items' in obj else obj
        cfg2=GKNConfig(n_layers=args.layers,n_filters=args.num_filters,walk_len=args.walk_len,gamma=args.gamma,embed_dim=args.embed_dim,drop_prob=args.dropout,use_layernorm=bool(args.use_layernorm),use_base=not bool(args.no_base),readout=args.readout,gate_tau=args.gate_tau,residual_cat=bool(args.residual_cat),max_motifs_per_type=args.max_motifs_per_type,motif_sample_ratio=args.motif_sample_ratio,recon_graph=args.recon_graph,recon_motif=args.recon_motif,lambda_motif=args.lambda_motif,recon_x=bool(args.recon_x),lambda_x=args.lambda_x,lambda_div=args.lambda_div,epochs=args.epochs,lr=args.lr,weight_decay=args.weight_decay,scheduler=args.scheduler,early_patience=args.early_patience,early_delta=args.early_delta,seed=args.seed,verbose=True)
        _,X=train_model(items,cfg2,device=args.device); torch.save({'X':X},emb_pt); print(f'[Step-02] embeddings -> {emb_pt}')
    rep_pt=str(rep_dir/f'representation_matrix_{args.tag}.pt')
    if args.skip_step03:
        if args.representation_cache and os.path.isfile(args.representation_cache): rep_pt=args.representation_cache
        else: raise FileNotFoundError(f'cached representation not found: {args.representation_cache}')
        print(f'[Step-03] skip -> {rep_pt}')
    else:
        print('\n[Step-03] linear self-representation'); emb_obj=torch.load(emb_pt,map_location='cpu'); X=emb_obj['X'] if isinstance(emb_obj,dict) and 'X' in emb_obj else emb_obj
        cfg3=LinearSRConfig(k=args.k,alpha=args.sr_alpha,beta=args.sr_beta,gamma=args.sr_gamma,lambda1=args.sr_lambda1,iters=args.sr_iters,tol=args.sr_tol,enforce_z_nonneg=bool(args.enforce_z_nonneg),enforce_z_colsum1=bool(args.enforce_z_colsum1),topk_sparsify=args.topk_sparsify,init_topk=args.init_topk,init_mutual_knn=bool(args.init_mutual_knn),symmetrize=args.symmetrize,normalize=args.normalize,verbose=True)
        A,Z,meta=linear_self_representation(X,cfg3); torch.save({'A':A,'Z':Z,'meta':meta},rep_pt); print(f'[Step-03] representation -> {rep_pt}')
    if not args.skip_step04:
        print('\n[Step-04] spectral clustering'); rep_obj=torch.load(rep_pt,map_location='cpu'); A=rep_obj['A'] if args.cluster_source=='A' else affinity_from_Z(rep_obj['Z'],normalize=args.normalize,symmetrize=args.symmetrize,self_loop=1.0); labels,U=spectral_cluster(A,k=args.k,n_init=args.kmeans_n_init,seed=args.cluster_seed,return_U=True)
        lab_path=clu_dir/f'labels_{args.tag}_{args.cluster_source}_seed{args.cluster_seed}.pt'; spec_path=clu_dir/f'spectral_embedding_{args.tag}_{args.cluster_source}_seed{args.cluster_seed}.pt'; torch.save({'labels':torch.as_tensor(labels,dtype=torch.long)},lab_path); torch.save({'U':torch.as_tensor(U,dtype=torch.float32)},spec_path)
        print(f'[Step-04] labels -> {lab_path}'); print(f'[Step-04] spectral embedding -> {spec_path}')
    print('\nDone.')
if __name__=='__main__': main()
