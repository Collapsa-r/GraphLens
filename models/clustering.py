# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

def _as_array(x)->np.ndarray:
    try:
        import torch
        if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    except Exception: pass
    if sp.issparse(x): x=x.toarray()
    return np.asarray(x,dtype=np.float64,order='C')

def _safe_symmetrize(M:np.ndarray,mode:str='avg')->np.ndarray:
    M=np.asarray(M,dtype=np.float64); np.fill_diagonal(M,0.0)
    if mode=='max': S=np.maximum(M,M.T)
    elif mode=='min': S=np.minimum(M,M.T)
    else: S=0.5*(M+M.T)
    np.fill_diagonal(S,0.0); return S

def _normalize_rows_or_cols(M:np.ndarray,how:str='none')->np.ndarray:
    if how=='none': return M
    eps=1e-12
    if how=='row':
        s=M.sum(axis=1,keepdims=True); s[s<eps]=1.0; return M/s
    if how=='col':
        s=M.sum(axis=0,keepdims=True); s[s<eps]=1.0; return M/s
    raise ValueError(f"normalize must be 'row' | 'col' | 'none', got {how!r}")

def affinity_from_Z(Z:csr_matrix|np.ndarray,symmetrize:str='avg',normalize:str='none',self_loop:float=1.0,clip_negative:bool=True,use_abs:bool=False,zscore:bool=False,scale_to_unit:bool=True,**kwargs)->csr_matrix:
    if 'sym' in kwargs and kwargs['sym'] is not None: symmetrize=kwargs.pop('sym')
    Z=_as_array(Z); Z=np.nan_to_num(Z,nan=0.0,posinf=0.0,neginf=0.0)
    if zscore:
        mu=Z.mean(axis=0,keepdims=True); sd=Z.std(axis=0,keepdims=True); sd[sd<1e-12]=1.0; Z=(Z-mu)/sd
    if use_abs: Z=np.abs(Z)
    if clip_negative: Z[Z<0.0]=0.0
    np.fill_diagonal(Z,0.0); Z=_normalize_rows_or_cols(Z,how=normalize); S=_safe_symmetrize(Z,mode=symmetrize)
    if self_loop is not None and self_loop>0.0: S=S+np.eye(S.shape[0],dtype=np.float64)*float(self_loop)
    if scale_to_unit:
        mx=S.max()
        if mx>0: S=S/mx
    S=sp.csr_matrix(S,dtype=np.float64); S.eliminate_zeros(); return S

@dataclass
class SpectralOptions:
    laplacian:str='sym'; knn:Optional[int]=None; eps_diag:float=1e-6; whiten:bool=True; kmeans_n_init:int=1000; random_state:int=42

def _to_spsym_affinity(A:sp.spmatrix|np.ndarray, eps:float)->csr_matrix:
    if sp.issparse(A):
        A=A.tocsr().astype(np.float64); A.data=np.abs(A.data); S=(A+A.T)*0.5; S.setdiag(0.0); S=S.tocsr()
        if S.nnz>0:
            m=float(S.data.max())
            if m>0: S.data/=m
        return S + sp.eye(S.shape[0],dtype=np.float64,format='csr')*eps
    A=np.asarray(A,dtype=np.float64); A=np.abs(A); S=0.5*(A+A.T); np.fill_diagonal(S,0.0); m=float(S.max())
    if m>0: S=S/m
    return sp.csr_matrix(S + np.eye(S.shape[0])*eps)

def _knn_graph(S:csr_matrix,k:Optional[int])->csr_matrix:
    if k is None or k<=0: return S.tocsr()
    S=S.tolil(copy=True); n=S.shape[0]
    for i in range(n):
        row=S.rows[i]; data=S.data[i]
        if len(row)>k:
            idx=np.argsort(data)[::-1][:k]; keep=set(row[j] for j in idx); new_rows=[]; new_data=[]
            for r,val in zip(row,data):
                if r in keep: new_rows.append(r); new_data.append(val)
            S.rows[i]=new_rows; S.data[i]=new_data
    S=(S.tocsr()+S.tocsr().T)*0.5; S.eliminate_zeros(); return S

def _build_laplacian(S:csr_matrix, mode:str)->csr_matrix:
    d=np.asarray(S.sum(axis=1)).ravel()
    if mode=='unnorm': return sp.diags(d)-S
    d[d<1e-12]=1e-12; D_inv_sqrt=sp.diags(1.0/np.sqrt(d)); S_norm=D_inv_sqrt@S@D_inv_sqrt; L_sym=sp.eye(S.shape[0],format='csr')-S_norm
    if mode=='sym': return L_sym
    if mode=='rw':
        D_inv=sp.diags(1.0/d); return sp.eye(S.shape[0],format='csr') - D_inv@S
    raise ValueError(f'Unknown laplacian mode: {mode!r}')

def _safe_smallest_eigvecs(L:csr_matrix,k:int)->np.ndarray:
    n=L.shape[0]; k=max(1,min(k,n-1)); Ld=L.toarray(); Ld=0.5*(Ld+Ld.T); Ld=Ld+1e-8*np.eye(n); w,U=np.linalg.eigh(Ld); return U[:, np.argsort(w)[:k]]

def _row_normalize(U:np.ndarray)->np.ndarray:
    nrm=np.linalg.norm(U,axis=1,keepdims=True); nrm[nrm<1e-12]=1.0; return U/nrm

def _spectral_embedding(A:sp.spmatrix|np.ndarray,k:int,laplacian:str='sym',knn:Optional[int]=None,eps_diag:float=1e-6,whiten:bool=True)->np.ndarray:
    S=_to_spsym_affinity(A,eps=eps_diag); S=_knn_graph(S,knn); L=_build_laplacian(S,mode=laplacian); U=_safe_smallest_eigvecs(L,k=k); return _row_normalize(U) if whiten else U

def spectral_cluster(A:sp.spmatrix|np.ndarray,k:int,n_init:int=300,seed:int=42,return_U:bool=False,laplacian:str='sym')->Tuple[np.ndarray, Optional[np.ndarray]]:
    U=_spectral_embedding(A,k=k,laplacian=laplacian,knn=None,eps_diag=1e-6,whiten=True)
    km=KMeans(n_clusters=int(k), n_init=max(int(n_init),50), random_state=int(seed), init='k-means++', verbose=0)
    labels=km.fit_predict(U)
    return (labels,U) if return_U else (labels,None)
