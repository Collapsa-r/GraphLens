# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import os
import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

def _ensure_dir(p:str):
    if p and not os.path.exists(p): os.makedirs(p,exist_ok=True)

def _to_numpy(x):
    if isinstance(x,np.ndarray): return x
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def _soft_shrink_dense(M,lam):
    if lam<=0: return M
    return np.sign(M)*np.maximum(np.abs(M)-lam,0.0)

def _apply_zero_diag_dense(M): np.fill_diagonal(M,0.0); return M

def _apply_nonneg_dense(M): M[M<0]=0.0; return M

def _col_stochastic_project_dense(M):
    colsum=M.sum(axis=0,keepdims=True); mask=np.abs(colsum)>1e-12
    if np.any(mask): M[:,mask.squeeze()] = M[:,mask.squeeze()]/colsum[:,mask.squeeze()]
    return M

def _csr_from_dense_keep_sparse(M,keep_topk=0,nonneg=False):
    if nonneg: M=_apply_nonneg_dense(M)
    if keep_topk and keep_topk>0:
        N=M.shape[0]; rows=[]; cols=[]; vals=[]
        for i in range(N):
            row=M[i]; nz=np.where(row!=0)[0]
            if nz.size==0: continue
            sel = nz[np.argpartition(np.abs(row[nz]), -keep_topk)[-keep_topk:]] if nz.size>keep_topk else nz
            rows.extend([i]*len(sel)); cols.extend(sel.tolist()); vals.extend(row[sel].tolist())
        return csr_matrix((vals,(rows,cols)),shape=M.shape)
    return csr_matrix(M)

def _build_knn_from_X(X,k):
    k=max(1,min(k,X.shape[0]-1)); nn=NearestNeighbors(n_neighbors=k,algorithm='auto',metric='euclidean').fit(X); _,idx=nn.kneighbors(X,return_distance=True); return idx

def _mutual_knn_lists(knn_idx,N):
    rev=[set() for _ in range(N)]
    for i in range(N):
        for j in knn_idx[i]: rev[j].add(i)
    return [np.array([j for j in knn_idx[i] if i in rev[j]],dtype=np.int64) for i in range(N)]

def _laplacian_normalized_from_W(W):
    W=W.tocsr(); W=0.5*(W+W.T); d=np.asarray(W.sum(axis=1)).ravel(); d[d<1e-12]=1e-12; D_inv_sqrt=diags(1.0/np.sqrt(d)); S=D_inv_sqrt@W@D_inv_sqrt; S=0.5*(S+S.T); return diags(np.ones(W.shape[0])) - S

def _safe_smallest_k_eigvecs(L,k):
    N=L.shape[0]; k=max(1,min(int(k),N-1))
    try:
        _,vecs=eigsh(L,k=k,which='SM',tol=1e-5,maxiter=5000); return vecs
    except Exception: pass
    try:
        _,vecs=eigsh(L,k=k,which='LM',sigma=0.0,tol=1e-5,maxiter=5000); return vecs
    except Exception: pass
    Ld=L.toarray() if issparse(L) else np.asarray(L); Ld=0.5*(Ld+Ld.T); Ld=Ld+1e-8*np.eye(N,dtype=Ld.dtype); w,v=np.linalg.eigh(Ld); return v[:, np.argsort(w)[:k]]

def _spectral_projector_S_from_C(C,k):
    W=0.5*(C+C.T); L=_laplacian_normalized_from_W(W); U=_safe_smallest_k_eigvecs(L,k=max(1,k)); return U@U.T

@dataclass
class LinearSRConfig:
    k:int=2; alpha:float=0.0; beta:float=3.0; gamma:float=0.0; lambda1:float=0.0; iters:int=60; tol:float=2e-4
    enforce_z_nonneg:bool=False; enforce_z_colsum1:bool=False; topk_sparsify:int=0; init_topk:int=24; init_mutual_knn:bool=True
    symmetrize:str='avg'; normalize:str='none'; verbose:bool=True

def linear_self_representation(X:np.ndarray,cfg:LinearSRConfig)->Tuple[csr_matrix, csr_matrix, Dict]:
    X=_to_numpy(X).astype(np.float32,copy=False); N,_=X.shape; G=X@X.T
    idx_knn=_build_knn_from_X(X,cfg.init_topk); knn_lists=_mutual_knn_lists(idx_knn,N) if cfg.init_mutual_knn else [idx_knn[i] for i in range(N)]
    Z0=np.zeros((N,N),dtype=np.float64); ridge_beta=max(1e-8,cfg.beta)
    for i in range(N):
        idx=knn_lists[i]
        if idx.size==0: continue
        Xi=X[i]; XN=X[idx]; Gm=XN@XN.T; Gm.flat[::Gm.shape[0]+1]+=ridge_beta; b=XN@Xi
        try: zi=np.linalg.solve(Gm,b)
        except np.linalg.LinAlgError: zi=np.linalg.lstsq(Gm,b,rcond=None)[0]
        Z0[i,idx]=zi
    _apply_zero_diag_dense(Z0)
    if cfg.enforce_z_nonneg: _apply_nonneg_dense(Z0)
    if cfg.lambda1>0: Z0=_soft_shrink_dense(Z0,cfg.lambda1)
    C=Z0.copy(); _apply_zero_diag_dense(C); _apply_nonneg_dense(C); C=_col_stochastic_project_dense(C)
    Z=Z0.copy(); I=np.eye(N,dtype=np.float64); GB=G+cfg.beta*I; RHS_const=(1.0+cfg.alpha)*G; prev_Z=Z.copy()
    for it in range(1,int(cfg.iters)+1):
        RHS=RHS_const + cfg.beta*C
        try: Z=np.linalg.solve(GB,RHS)
        except np.linalg.LinAlgError: Z=np.linalg.lstsq(GB,RHS,rcond=None)[0]
        _apply_zero_diag_dense(Z)
        if cfg.lambda1>0: Z=_soft_shrink_dense(Z,cfg.lambda1)
        if cfg.enforce_z_nonneg: _apply_nonneg_dense(Z)
        if cfg.enforce_z_colsum1: Z=_col_stochastic_project_dense(Z)
        if cfg.topk_sparsify>0:
            for i in range(N):
                row=Z[i]; nz=np.where(row!=0)[0]
                if nz.size>cfg.topk_sparsify:
                    keep=nz[np.argpartition(np.abs(row[nz]), -cfg.topk_sparsify)[-cfg.topk_sparsify:]]; mask=np.ones(N,dtype=bool); mask[keep]=False; Z[i,mask]=0.0
        C=np.maximum(Z,0.0); _apply_zero_diag_dense(C); C=_col_stochastic_project_dense(C)
        if cfg.gamma>0:
            S=_spectral_projector_S_from_C(csr_matrix(C),k=cfg.k); Lc=np.diag(C.sum(axis=1))-C; C=C-cfg.gamma*(Lc@S); C=np.maximum(C,0.0); _apply_zero_diag_dense(C); C=_col_stochastic_project_dense(C)
        rel=np.linalg.norm(Z-prev_Z)/(np.linalg.norm(prev_Z)+1e-12)
        if cfg.verbose: print(f'[LSR] iter={it:03d} rel_change={rel:.6e}')
        if rel<cfg.tol: break
        prev_Z=Z.copy()
    W=Z.copy()
    if cfg.symmetrize=='avg': W=0.5*(W+W.T)
    elif cfg.symmetrize=='max': W=np.maximum(W,W.T)
    elif cfg.symmetrize=='min': W=np.minimum(W,W.T)
    W=np.maximum(W,0.0); _apply_zero_diag_dense(W)
    if cfg.normalize=='row':
        rs=W.sum(axis=1,keepdims=True); rs[rs==0]=1.0; W=W/rs
    elif cfg.normalize=='col':
        cs=W.sum(axis=0,keepdims=True); cs[cs==0]=1.0; W=W/cs
    elif cfg.normalize!='none': raise ValueError("normalize must be one of {'none','row','col'}")
    A=csr_matrix(W); Z_out=_csr_from_dense_keep_sparse(Z,keep_topk=0,nonneg=False); meta={'algo':'altmin-closedform', **asdict(cfg)}
    return A,Z_out,meta

def save_representation(A:csr_matrix, Z:csr_matrix, out_dir:str, tag:str, meta:Dict)->str:
    _ensure_dir(out_dir); out_pt=os.path.join(out_dir,f'representation_matrix_{tag}.pt'); torch.save({'A':A,'Z':Z,'meta':meta},out_pt); return out_pt
