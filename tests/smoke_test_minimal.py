import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.graph_kernel_network import GKNConfig, train_model
from models.self_representation import LinearSRConfig, linear_self_representation
from models.clustering import affinity_from_Z, spectral_cluster


def main():
    items = [
        {'graph_id': 0, 'edge_index': torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long), 'num_nodes': 3, 'x': torch.ones(3, 1), 'motifs': {'triangle': [(0, 1, 2)]}},
        {'graph_id': 1, 'edge_index': torch.tensor([[0, 1], [1, 2]], dtype=torch.long), 'num_nodes': 3, 'x': torch.ones(3, 1), 'motifs': {'wedge': [(0, 1, 2)]}},
        {'graph_id': 2, 'edge_index': torch.tensor([[0], [1]], dtype=torch.long), 'num_nodes': 2, 'x': torch.ones(2, 1), 'motifs': {'edge': [(0, 1)]}},
    ]
    _, X = train_model(items, GKNConfig(epochs=1, verbose=False, n_layers=1, n_filters=2, embed_dim=8, readout='last', residual_cat=True), device='cpu')
    _, Z, _ = linear_self_representation(X.numpy(), LinearSRConfig(k=2, iters=2, verbose=False, init_topk=2))
    A = affinity_from_Z(Z)
    labels, _ = spectral_cluster(A, k=2, return_U=True)
    print('smoke test passed:', labels.tolist())


if __name__ == '__main__':
    main()
