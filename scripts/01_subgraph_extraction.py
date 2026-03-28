#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.subgraph_extraction import ExtractConfig, extract_dataset_to_cache

def main():
    ap=argparse.ArgumentParser('01: TU dataset -> motif cache')
    ap.add_argument('--tu_dataset',type=str,required=True)
    ap.add_argument('--tag',type=str,default='default')
    ap.add_argument('--data_root',type=str,default='./data/TUDataset')
    ap.add_argument('--out_root',type=str,default='./outputs')
    ap.add_argument('--sample_manifest',type=str,default=None)
    ap.add_argument('--motifs',type=str,default=None)
    ap.add_argument('--z_threshold',type=float,default=None)
    ap.add_argument('--null_iters',type=int,default=None)
    ap.add_argument('--edge_swaps_per_edge',type=float,default=None)
    ap.add_argument('--pairwise_max_dist',type=int,default=None)
    ap.add_argument('--min_keep_per_type',type=int,default=None)
    args=ap.parse_args()
    cfg=ExtractConfig(data_root=args.data_root,out_root=args.out_root)
    if args.motifs is not None: cfg.motifs=tuple(s.strip() for s in args.motifs.split(',') if s.strip())
    if args.z_threshold is not None: cfg.z_threshold=float(args.z_threshold)
    if args.null_iters is not None: cfg.null_iters=int(args.null_iters)
    if args.edge_swaps_per_edge is not None: cfg.edge_swaps_per_edge=float(args.edge_swaps_per_edge)
    if args.pairwise_max_dist is not None: cfg.pairwise_max_dist=int(args.pairwise_max_dist)
    if args.min_keep_per_type is not None: cfg.min_keep_per_type=int(args.min_keep_per_type)
    ret=extract_dataset_to_cache(dataset_name=args.tu_dataset, tag=args.tag, cfg=cfg, sample_manifest=args.sample_manifest)
    print('\n[01] motif extraction done'); print(' - Cache    :', ret['cache_path']); print(' - Manifest :', ret['manifest_path']); print(' - #Graphs  :', ret['N'])
if __name__=='__main__': main()
