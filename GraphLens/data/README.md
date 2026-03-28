# Data Directory

This repository does **not** bundle third-party datasets.

Expected layout for TU datasets:

```text
data/
└── TUDataset/
```

`Step-01` downloads or reads TU datasets through `torch-geometric`.
If you evaluate on your own data, place the processed data in a compatible format
and adapt the loader if needed.
