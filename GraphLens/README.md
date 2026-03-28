# Minimal Experimental Codebase

A cleaned research-code artifact for experiment release.
This repository keeps only the **core runnable pipeline** and removes all visualization code.

## What is included

- motif/subgraph extraction
- graph kernel network training
- linear self-representation
- spectral clustering
- an end-to-end pipeline script
- example commands for Linux and PowerShell
- artifact-oriented metadata files (`LICENSE`, `CITATION.cff`, `environment.yml`)

## Repository Layout

```text
graph_experiment_core/
├── models/
│   ├── subgraph_extraction.py
│   ├── graph_kernel_network.py
│   ├── self_representation.py
│   └── clustering.py
├── scripts/
│   ├── 01_subgraph_extraction.py
│   ├── 02_train_rwk.py
│   ├── 03_self_representation.py
│   └── 04_clustering.py
├── examples/
│   ├── run_imdb_binary.sh
│   └── run_imdb_binary.ps1
├── tests/
│   └── smoke_test_minimal.py
├── data/
│   └── README.md
├── docs/
│   └── ARTIFACT_AVAILABILITY_TEMPLATE.md
├── run_pipeline.py
├── requirements.txt
├── environment.yml
├── LICENSE
└── CITATION.cff
```

## Install

### Pip

```bash
pip install -r requirements.txt
```

### Conda

```bash
conda env create -f environment.yml
conda activate graph-exp-core
```

## Quick Start

### Step-by-step

```bash
python scripts/01_subgraph_extraction.py   --tu_dataset IMDB-BINARY   --tag imdbb

python scripts/02_train_rwk.py   --subgraphs ./outputs/cache/subgraphs_IMDB-BINARY_imdbb.pt   --tag imdbb

python scripts/03_self_representation.py   --emb ./outputs/embeddings/emb_imdbb.pt   --tag imdbb   --k 2

python scripts/04_clustering.py   --rep ./outputs/representation_matrices/representation_matrix_imdbb.pt   --tag imdbb   --k 2   --source Z
```

### End-to-end

```bash
python run_pipeline.py   --dataset IMDB-BINARY   --tag imdbb   --k 2
```

## Reproducibility Notes

- All plotting and figure-generation code has been removed.
- The repository is intentionally focused on the experiment path needed to run the method.
- For graphs where no motif survives filtering, the code falls back to using edges as the simplest motif so the pipeline can still run.
- `Step-01` requires `torch-geometric` because TU datasets are loaded through PyG.
- Third-party datasets are not bundled in this repository.

## Suggested Artifact Sentence for the Paper

Use the template in `docs/ARTIFACT_AVAILABILITY_TEMPLATE.md` and replace the placeholder repository URL.
