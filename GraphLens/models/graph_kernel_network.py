# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _row_normalize(A: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    d=A.sum(-1,keepdim=True).clamp_min(eps)
    return A/d

def infer_num_nodes_from_item(item: Dict[str,Any]) -> int:
    if 'num_nodes' in item: return int(item['num_nodes'])
    ei=item.get('edge_index',None)
    if isinstance(ei,torch.Tensor) and ei.numel()>0: return int(torch.max(ei).item())+1
    x=item.get('x',None)
    if isinstance(x,torch.Tensor): return int(x.size(0))
    return 1

def _build_dense_A(item: Dict[str,Any], n:int, device:str) -> torch.Tensor:
    A=torch.zeros((n,n),dtype=torch.float32,device=device)
    ei=item.get('edge_index',None)
    if isinstance(ei,torch.Tensor) and ei.numel()>0: A[ei[0].long(), ei[1].long()] = 1.0
    A=0.5*(A+A.t()); A.fill_diagonal_(0.0)
    return A

def _degree_channel(item: Dict[str,Any], n:int, device:str) -> torch.Tensor:
    deg=torch.zeros(n,1,device=device); ei=item.get('edge_index',None)
    if isinstance(ei,torch.Tensor) and ei.numel()>0:
        src,dst=ei[0].to(device), ei[1].to(device); one=torch.ones(src.size(0),1,device=device)
        deg.index_add_(0,src,one); deg.index_add_(0,dst,one)
    return deg

def _motif_template_matrix(name:str, size:int)->np.ndarray:
    name=(name or '').lower(); M=np.zeros((size,size),dtype=np.float32)
    def add_edges(edges):
        for u,v in edges:
            if 0<=u<size and 0<=v<size and u!=v:
                M[u,v]=1.0; M[v,u]=1.0
    if name in ('triangle','clique3'): add_edges([(0,1),(1,2),(0,2)])
    elif name in ('wedge','open_triad','v'): add_edges([(0,1),(1,2)])
    elif name=='edge': add_edges([(0,1)])
    elif name.startswith('path'): add_edges([(i,i+1) for i in range(size-1)])
    elif name in ('cycle4','square'): add_edges([(0,1),(1,2),(2,3),(3,0)])
    elif name=='cycle5': add_edges([(i,(i+1)%5) for i in range(5)])
    elif name=='cycle6': add_edges([(i,(i+1)%size) for i in range(size)])
    elif name=='diamond': add_edges([(0,1),(0,2),(1,2),(0,3),(1,3)])
    elif name=='clique4': add_edges([(i,j) for i in range(4) for j in range(i+1,4)])
    elif name.startswith('star'): add_edges([(0,i) for i in range(1,size)])
    elif name in ('tailed_triangle','triangle_tail'): add_edges([(0,1),(1,2),(0,2),(2,3)])
    else:
        if size>=3: add_edges([(i,(i+1)%size) for i in range(size)])
        else: add_edges([(0,1)])
    np.fill_diagonal(M,0.0)
    return M

@dataclass
class GKNConfig:
    n_layers:int=2; n_filters:int=8; walk_len:int=3; gamma:float=0.7; embed_dim:int=128; drop_prob:float=0.10
    use_layernorm:bool=True; readout:str='concat'; residual_cat:bool=True; gate_tau:float=1.0; use_base:bool=True
    max_motifs_per_type:int=-1; motif_sample_ratio:float=1.0
    recon_graph:str='mse'; recon_motif:str='off'; lambda_motif:float=1.0; recon_x:bool=False; lambda_x:float=1e-3; lambda_div:float=3e-4
    epochs:int=80; lr:float=2e-3; weight_decay:float=5e-4; scheduler:str='plateau'; early_patience:int=20; early_delta:float=5e-4
    seed:int=0; verbose:bool=True; batch_log_every:int=50

class LearnableFilter(nn.Module):
    def __init__(self,size:int,motif_name:str|None=None):
        super().__init__(); init=_motif_template_matrix(motif_name or '', size); init=torch.from_numpy(init).float()+0.05*torch.randn(size,size); self.W_adj=nn.Parameter(init)
    def adj(self)->torch.Tensor:
        W=F.softplus(self.W_adj); W=0.5*(W+W.t()); W.fill_diagonal_(0.0); return W

def rwk_struct_from_TM_SF(Tm:torch.Tensor, Sf:torch.Tensor, L:int, gamma:float, length_gates:Optional[torch.Tensor]=None)->torch.Tensor:
    Km=torch.kron(Tm,Sf); acc=torch.tensor(0.0,device=Tm.device); M=torch.eye(Km.size(0),device=Tm.device)
    for p in range(1,L+1):
        M=M@Km; coeff=gamma**p
        if length_gates is not None: coeff=coeff*F.softplus(length_gates[p-1])
        acc=acc+coeff*torch.trace(M)
    return acc

class MotifResponseLayer(nn.Module):
    def __init__(self,motif_sizes:Dict[str,int],n_filters:int,feat_dim_in:int,d_out:int,walk_len:int,gamma:float,gate_tau:float=1.0,use_layernorm:bool=True,drop_prob:float=0.1,residual_cat:bool=True,max_motifs_per_type:int=-1,motif_sample_ratio:float=1.0):
        super().__init__(); self.types=list(motif_sizes.keys()); self.walk_len=walk_len; self.gamma=gamma; self.gate_tau=gate_tau; self.length_gates=nn.Parameter(torch.zeros(walk_len)); self.n_filters=n_filters; self.residual_cat=residual_cat; self.max_motifs_per_type=int(max_motifs_per_type); self.motif_sample_ratio=float(motif_sample_ratio)
        self.filters=nn.ModuleDict({t: nn.ModuleList([LearnableFilter(size=motif_sizes[t], motif_name=t) for _ in range(n_filters)]) for t in self.types})
        in_ch=len(self.types)*n_filters + (feat_dim_in if residual_cat else 0); self.in_ch=in_ch; self.norm=nn.LayerNorm(in_ch) if use_layernorm else nn.Identity(); hid=max(d_out,64)
        self.encoder=nn.Sequential(nn.Linear(in_ch,hid), nn.GELU(), nn.Dropout(drop_prob), nn.Linear(hid,d_out))
    def _maybe_sample_instances(self,instances):
        inst=list(instances or [])
        if not inst: return inst
        if 0.0 < self.motif_sample_ratio < 1.0:
            target=max(1,int(round(len(inst)*self.motif_sample_ratio)))
            if target < len(inst):
                step=max(1, len(inst)//target); inst=inst[::step][:target]
        if self.max_motifs_per_type>0 and len(inst)>self.max_motifs_per_type:
            step=max(1, len(inst)//self.max_motifs_per_type); inst=inst[::step][:self.max_motifs_per_type]
        return inst
    def forward(self,A_full:torch.Tensor,motifs:Dict[str,List[Tuple[int,...]]],H_in:Optional[torch.Tensor]=None)->torch.Tensor:
        device=A_full.device; n=A_full.size(0); Phi=torch.zeros(n,self.in_ch,device=device); ch=0
        for t in self.types:
            inst=self._maybe_sample_instances((motifs or {}).get(t,[]) or []); Sfs=[_row_normalize(flt.adj()) for flt in self.filters[t]]; Hm=torch.zeros(n,self.n_filters,device=device)
            if inst:
                for nodes in inst:
                    nodes=list(nodes); As=A_full[nodes][:,nodes]; Tm=_row_normalize(As)
                    sims=[rwk_struct_from_TM_SF(Tm,Sf,L=self.walk_len,gamma=self.gamma,length_gates=self.length_gates) for Sf in Sfs]
                    sims=torch.stack(sims,dim=0); alphas=F.softmax(sims/max(1e-6,self.gate_tau),dim=0); contrib=alphas*sims
                    for j in range(self.n_filters): Hm[nodes,j]+=contrib[j]/float(len(nodes))
            Phi[:,ch:ch+self.n_filters]=Hm; ch+=self.n_filters
        if self.residual_cat and H_in is not None: Phi[:,ch:ch+H_in.size(1)] = H_in
        return self.encoder(self.norm(Phi))

class GraphKernelNetwork(nn.Module):
    def __init__(self,motif_sizes:Dict[str,int],cfg:GKNConfig):
        super().__init__(); self.cfg=cfg; base_dim=1 if cfg.use_base else 0; d_in=max(base_dim,1); layers=[]
        for _ in range(cfg.n_layers):
            layers.append(MotifResponseLayer(motif_sizes=motif_sizes,n_filters=cfg.n_filters,feat_dim_in=d_in,d_out=cfg.embed_dim,walk_len=cfg.walk_len,gamma=cfg.gamma,gate_tau=cfg.gate_tau,use_layernorm=cfg.use_layernorm,drop_prob=cfg.drop_prob,residual_cat=cfg.residual_cat,max_motifs_per_type=cfg.max_motifs_per_type,motif_sample_ratio=cfg.motif_sample_ratio))
            d_in = cfg.embed_dim if not cfg.residual_cat else (cfg.embed_dim + d_in)
        self.layers=nn.ModuleList(layers); self.x_decoder=nn.Linear(cfg.embed_dim,1)
    def _graph_recon(self,H,A,kind):
        if kind=='off': return torch.tensor(0.0,device=A.device)
        S=H@H.t()
        if kind=='bce':
            target=A; pos=target.sum().item(); neg=target.numel()-pos-A.size(0); posw=(neg/max(pos,1.0)) if pos>0 else 1.0
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(posw,device=A.device))(S,target)
        A_hat=torch.sigmoid(S); A_hat=0.5*(A_hat+A_hat.t()); A_hat.fill_diagonal_(0.0); return F.mse_loss(A_hat,A)
    def _motif_recon(self,H,A,motifs,kind):
        if kind=='off': return torch.tensor(0.0,device=A.device)
        loss=torch.tensor(0.0,device=A.device); total=0
        for inst in (motifs or {}).values():
            for nodes in inst:
                nodes=list(nodes); S=H[nodes]@H[nodes].t()
                if kind=='bce': loss=loss+F.binary_cross_entropy_with_logits(S,A[nodes][:,nodes])
                else:
                    A_hat=torch.sigmoid(S); A_hat=0.5*(A_hat+A_hat.t()); A_hat.fill_diagonal_(0.0); loss=loss+F.mse_loss(A_hat,A[nodes][:,nodes])
                total+=1
        return loss/max(1,total)
    def _filter_diversity(self):
        loss=torch.tensor(0.0,device=self.x_decoder.weight.device)
        for layer in self.layers:
            for _,bank in layer.filters.items():
                mats=[flt.adj() for flt in bank]
                for i in range(len(mats)):
                    for j in range(i+1,len(mats)): loss=loss+F.mse_loss(mats[i],mats[j])
        return loss
    def forward_one(self,item:Dict[str,Any],device:str='cuda'):
        A=_build_dense_A(item,infer_num_nodes_from_item(item),device); n=A.size(0); x0=_degree_channel(item,n,device) if self.cfg.use_base else torch.zeros(n,1,device=device); motifs=item.get('motifs',{}) or {}; H=x0
        for layer in self.layers: H=layer(A,motifs,H_in=(H if self.cfg.residual_cat else None))
        if self.cfg.readout=='concat': emb=torch.cat([H.mean(0), H.max(0).values], dim=0)
        elif self.cfg.readout=='sum': emb=H.sum(0)
        else: emb=H.mean(0)
        Lg=self._graph_recon(H,A,self.cfg.recon_graph); Lm=self._motif_recon(H,A,motifs,self.cfg.recon_motif); Ld=self._filter_diversity()*self.cfg.lambda_div; Lx=torch.tensor(0.0,device=A.device)
        if self.cfg.recon_x:
            x_hat=self.x_decoder(H); Lx=F.mse_loss(x_hat,x0)*self.cfg.lambda_x
        loss = Lg + self.cfg.lambda_motif*Lm + Ld + Lx
        return {'emb': emb, 'loss': loss, 'loss_graph': Lg, 'loss_motif': Lm, 'loss_div': Ld, 'loss_x': Lx}

def _collect_motif_sizes(items):
    sizes={}
    for item in items:
        for name,inst in (item.get('motifs',{}) or {}).items():
            if inst: sizes[name]=max(sizes.get(name,0), len(inst[0]))
    if not sizes:
        sizes['edge']=2
        for item in items:
            ei=item.get('edge_index')
            if isinstance(ei,torch.Tensor) and ei.numel()>0: edges=[(int(u),int(v)) for u,v in zip(ei[0].tolist(), ei[1].tolist()) if int(u)<int(v)]
            else: edges=[]
            item['motifs']={'edge': edges}
    return sizes

def train_model(items:List[Dict[str,Any]], cfg:GKNConfig, device:str='cuda'):
    torch.manual_seed(int(cfg.seed)); np.random.seed(int(cfg.seed)); random.seed(int(cfg.seed))
    if device=='cuda' and not torch.cuda.is_available(): device='cpu'
    motif_sizes=_collect_motif_sizes(items); model=GraphKernelNetwork(motif_sizes,cfg).to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    if cfg.scheduler=='plateau': sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=8)
    elif cfg.scheduler=='cosine': sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.epochs)
    else: sch=None
    best_loss=float('inf'); wait=0
    for ep in range(1,cfg.epochs+1):
        model.train(); running=0.0
        for bi,item in enumerate(items,1):
            out=model.forward_one(item,device=device); loss=out['loss']; opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); running += float(loss.detach().cpu().item())
            if cfg.verbose and bi % max(1,cfg.batch_log_every)==0: print(f'[GKN] epoch {ep:03d} batch {bi:04d} loss={loss.item():.4f}')
        epoch_loss=running/max(1,len(items))
        if cfg.verbose: print(f'[GKN] epoch {ep:03d} mean_loss={epoch_loss:.6f}')
        if sch is not None:
            if cfg.scheduler=='plateau': sch.step(epoch_loss)
            else: sch.step()
        if epoch_loss + cfg.early_delta < best_loss: best_loss=epoch_loss; wait=0
        else: wait+=1
        if wait >= cfg.early_patience:
            if cfg.verbose: print(f'[GKN] early stop at epoch {ep}')
            break
    model.eval()
    with torch.no_grad():
        X=[]
        for item in items:
            out=model.forward_one(item,device=device); X.append(out['emb'].detach().cpu())
        X=torch.stack(X,dim=0).float()
    return model,X
