# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import itertools, json, random
import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

@dataclass
class ExtractConfig:
    data_root: str = './data/TUDataset'
    out_root: str = './outputs'
    motifs: Tuple[str, ...] = ('triangle','wedge','path3','cycle4','cycle5','diamond','clique4','star4','tailed_triangle')
    z_threshold: float = 1.0
    null_iters: int = 20
    edge_swaps_per_edge: float = 5.0
    random_seed: int = 0
    pairwise_max_dist: int = 2
    min_keep_per_type: int = 0
    fallback_to_edges: bool = True

def _to_simple_undirected(G: nx.Graph) -> nx.Graph:
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        H = nx.Graph(); H.add_nodes_from(G.nodes())
        for u, v in G.edges():
            if u != v: H.add_edge(u, v)
        G = H
    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def _enumerate_triangles(G):
    tri=set()
    for u in G.nodes():
        Nu=set(G.neighbors(u))
        for v,w in itertools.combinations(Nu,2):
            if G.has_edge(v,w): tri.add(tuple(sorted((u,v,w))))
    return sorted(tri)

def _enumerate_wedge(G):
    out=[]; seen=set()
    for c in G.nodes():
        N=list(G.neighbors(c))
        if len(N)<2: continue
        for a,b in itertools.combinations(N,2):
            if G.has_edge(a,b): continue
            key=tuple(sorted((a,c,b)))
            if key not in seen:
                seen.add(key); out.append(key)
    return out

def _enumerate_path3(G):
    res=[]; seen=set()
    for b,c in G.edges():
        for a in (set(G.neighbors(b))-{c}):
            for d in (set(G.neighbors(c))-{b}):
                if a==d: continue
                if G.has_edge(a,c) or G.has_edge(b,d) or G.has_edge(a,d): continue
                cand=None
                if G.has_edge(a,b) and G.has_edge(c,d): cand=(a,b,c,d)
                if cand is None:
                    for x1,x2,x3,x4 in ((a,b,c,d),(a,c,b,d)):
                        if G.has_edge(x1,x2) and G.has_edge(x2,x3) and G.has_edge(x3,x4):
                            cand=(x1,x2,x3,x4); break
                if cand is None: continue
                key=tuple(sorted((cand[0],cand[3])))+tuple(sorted((cand[1],cand[2])))
                if key not in seen:
                    seen.add(key); res.append(cand)
    return res

def _enumerate_cycle4(G):
    cyc=set(); nodes=list(G.nodes())
    for u,v in itertools.combinations(nodes,2):
        if G.has_edge(u,v): continue
        commons=list(set(G.neighbors(u)) & set(G.neighbors(v)))
        if len(commons)>=2:
            for a,b in itertools.combinations(commons,2):
                cyc.add(tuple(sorted((u,a,v,b))))
    return sorted(cyc)

def _enumerate_cycle_k(G,k):
    out=set(); nodes=sorted(G.nodes())
    for s in nodes:
        stack=[(s,[s])]
        while stack:
            cur,path=stack.pop()
            if len(path)==k:
                if G.has_edge(cur,s) and min(path)==s: out.add(tuple(sorted(path)))
                continue
            for nb in G.neighbors(cur):
                if nb==s and len(path)<k: continue
                if nb in path: continue
                ok=True
                if len(path)>=2:
                    for p in path[:-2]:
                        if G.has_edge(nb,p): ok=False; break
                if ok: stack.append((nb,path+[nb]))
    return sorted(out)

def _enumerate_diamond(G):
    diamonds=set()
    for u,v in G.edges():
        commons=list((set(G.neighbors(u))-{v}) & (set(G.neighbors(v))-{u}))
        if len(commons)>=2:
            for a,b in itertools.combinations(commons,2):
                if not G.has_edge(a,b): diamonds.add(tuple(sorted((u,v,a,b))))
    return sorted(diamonds)

def _enumerate_clique4(G):
    return sorted(tuple(sorted(c)) for c in nx.find_cliques(G) if len(c)==4)

def _enumerate_star4(G):
    out=[]; seen=set()
    for v in G.nodes():
        N=list(G.neighbors(v))
        if len(N)<3: continue
        for a,b,c in itertools.combinations(N,3):
            if G.has_edge(a,b) or G.has_edge(a,c) or G.has_edge(b,c): continue
            key=tuple(sorted((v,a,b,c)))
            if key not in seen:
                seen.add(key); out.append(key)
    return out

def _enumerate_tailed_triangle(G):
    out=[]; seen=set()
    for u,v,w in _enumerate_triangles(G):
        tri={u,v,w}
        for t in tri:
            others=list(tri-{t})
            for x in G.neighbors(t):
                if x in tri: continue
                if not G.has_edge(x,others[0]) and not G.has_edge(x,others[1]):
                    key=tuple(sorted((u,v,w,x)))
                    if key not in seen:
                        seen.add(key); out.append(key)
    return out

_MOTIF_ENUM_FN = {
    'triangle': _enumerate_triangles, 'wedge': _enumerate_wedge, 'path3': _enumerate_path3,
    'cycle4': _enumerate_cycle4, 'cycle5': lambda G: _enumerate_cycle_k(G,5), 'cycle6': lambda G: _enumerate_cycle_k(G,6),
    'diamond': _enumerate_diamond, 'clique4': _enumerate_clique4, 'star4': _enumerate_star4, 'tailed_triangle': _enumerate_tailed_triangle,
}

def _pairwise_distances_within_k(G,nodes,k):
    nodes=list(nodes)
    for i in range(len(nodes)):
        sp=nx.single_source_shortest_path_length(G,nodes[i],cutoff=k)
        for j in range(i+1,len(nodes)):
            if nodes[j] not in sp: return False
    return True

def _count_motifs(G,motifs,k_pairwise=2):
    per_k={'path3':3,'cycle6':3}
    res={}
    for name in motifs:
        if name not in _MOTIF_ENUM_FN: continue
        tuples=_MOTIF_ENUM_FN[name](G)
        need_k=per_k.get(name,k_pairwise)
        res[name]=[t for t in tuples if _pairwise_distances_within_k(G,t,need_k)]
    return res

def _motif_stats(G,cfg):
    motif_sets=_count_motifs(G,cfg.motifs,k_pairwise=cfg.pairwise_max_dist)
    return {k: len(v) for k,v in motif_sets.items()}

def _double_edge_swap_preserve_degree(G, swaps, seed):
    H=G.copy()
    try: nx.double_edge_swap(H, nswap=swaps, max_tries=swaps*10, seed=seed)
    except Exception: pass
    return H

def _zscore_from_null(G,cfg):
    obs=_motif_stats(G,cfg)
    if G.number_of_edges()==0 or cfg.null_iters<=0:
        return {k:(float(obs.get(k,0)), float(obs.get(k,0)), 0.0) for k in cfg.motifs}
    E=G.number_of_edges(); swaps=max(1,int(cfg.edge_swaps_per_edge*E)); rng=random.Random(cfg.random_seed)
    samples={k:[] for k in cfg.motifs}
    for _ in range(cfg.null_iters):
        H=_double_edge_swap_preserve_degree(G,swaps=swaps,seed=rng.randint(1,10**9))
        cnts=_motif_stats(H,cfg)
        for k in cfg.motifs: samples[k].append(cnts.get(k,0))
    stats={}
    for k in cfg.motifs:
        mu=float(np.mean(samples[k])) if samples[k] else 0.0
        sigma=float(np.std(samples[k],ddof=1)) if len(samples[k])>1 else 1e-8
        z=(float(obs.get(k,0))-mu)/(sigma+1e-8)
        stats[k]=(float(obs.get(k,0)),mu,z)
    return stats

def _filter_motifs_by_zscore(G,cfg):
    motif_sets=_count_motifs(G,cfg.motifs,k_pairwise=cfg.pairwise_max_dist)
    if cfg.z_threshold<=0:
        keep=motif_sets
    else:
        keep={}; stats=_zscore_from_null(G,cfg)
        for name,tuples in motif_sets.items():
            if stats.get(name,(0.0,0.0,-1e9))[2] >= cfg.z_threshold:
                keep[name]=tuples
            elif cfg.min_keep_per_type>0:
                keep[name]=tuples[:cfg.min_keep_per_type]
            else:
                keep[name]=[]
    if cfg.fallback_to_edges and sum(len(v) for v in keep.values())==0:
        keep={'edge':[tuple(sorted(e)) for e in G.edges()]}
    return keep

def _ensure_x(data):
    x=getattr(data,'x',None)
    if x is not None:
        if x.dim()==1: x=x.view(-1,1)
        return x.float()
    n=int(data.num_nodes); deg=torch.zeros(n,dtype=torch.float32); ei=data.edge_index
    if ei.numel()>0:
        deg.index_add_(0,ei[0],torch.ones(ei.size(1),dtype=torch.float32))
        deg.index_add_(0,ei[1],torch.ones(ei.size(1),dtype=torch.float32))
    return deg.view(-1,1)

def extract_dataset_to_cache(dataset_name: str, tag: str='default', cfg: Optional[ExtractConfig]=None, sample_manifest: Optional[str]=None) -> Dict[str, Any]:
    cfg = cfg or ExtractConfig()
    out_root=Path(cfg.out_root); cache_dir=out_root/'cache'; ingest_dir=out_root/'ingest'/'subsets'; meta_dir=out_root/'metadata'
    for d in (cache_dir, ingest_dir, meta_dir): d.mkdir(parents=True, exist_ok=True)
    manifest_path=Path(sample_manifest) if sample_manifest else ingest_dir/f'tud_{dataset_name}_{tag}.pt'
    ds_full=TUDataset(root=cfg.data_root, name=dataset_name)
    if manifest_path.is_file():
        mani=torch.load(manifest_path,map_location='cpu')
        if 'graph_ids' in mani: sel_ids=torch.as_tensor(mani['graph_ids']).long().tolist()
        elif 'indices' in mani: sel_ids=torch.as_tensor(mani['indices']).long().tolist()
        else: sel_ids=list(range(len(ds_full)))
    else:
        sel_ids=list(range(len(ds_full)))
    items=[]; labels=[]; meta_records=[]
    for gid in sel_ids:
        data=ds_full[int(gid)]
        G=_to_simple_undirected(to_networkx(data,to_undirected=True))
        motif_sets=_filter_motifs_by_zscore(G,cfg)
        edge_index=torch.tensor(list(G.edges()),dtype=torch.long).t().contiguous() if G.number_of_edges()>0 else torch.empty(2,0,dtype=torch.long)
        items.append({'graph_id': int(gid), 'edge_index': edge_index, 'num_nodes': int(G.number_of_nodes()), 'x': _ensure_x(data), 'motifs': motif_sets})
        y=int(data.y.item()) if getattr(data,'y',None) is not None else -1
        labels.append(max(y,0))
        meta_records.append({'graph_id':int(gid),'num_nodes':int(G.number_of_nodes()),'num_edges':int(G.number_of_edges()),'motif_counts':{k:len(v) for k,v in motif_sets.items()}})
    cache_path=cache_dir/f'subgraphs_{dataset_name}_{tag}.pt'
    torch.save({'items': items, 'dataset': dataset_name, 'mode': 'small'}, cache_path)
    torch.save({'task': 'graph', 'dataset': dataset_name, 'graph_ids': torch.as_tensor([it['graph_id'] for it in items],dtype=torch.long), 'labels': torch.as_tensor(labels,dtype=torch.long)}, manifest_path)
    with open(meta_dir/f'{dataset_name}_{tag}_summary.json','w',encoding='utf-8') as f:
        json.dump({'dataset':dataset_name,'tag':tag,'num_graphs':len(items),'graphs':meta_records},f,ensure_ascii=False,indent=2)
    return {'items': items, 'cache_path': str(cache_path), 'subgraphs_pt': str(cache_path), 'manifest_path': str(manifest_path), 'N': len(items)}
