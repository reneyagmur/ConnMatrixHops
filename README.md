# ConnMatrixHops

Visualize signal flow through N-hop subcircuits in connectivity matrices.

ConnMatrixHops performs BFS-style traversal from seed neurons through a directed connectivity graph, producing multi-panel heatmap visualizations that reveal how signals propagate hop by hop.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from connmatrixhops import MatrixAnalyzer

# From an edge list
edges = pd.read_csv("data/toy_data/edge_list.csv")
clusters = pd.read_csv("data/toy_data/cluster_assignments.csv")

analyzer = MatrixAnalyzer(
    edge_list=edges,
    source_col="pre_id",
    target_col="post_id",
    weight_col="synapse_size",
    metadata=clusters,
    cell_id_col="cell_id",
    cluster_col="cluster",
)

# Plot 4-hop signal flow from a cluster
analyzer.plot_flow("A", n_steps=4)
```

```python
import numpy as np
from connmatrixhops import MatrixAnalyzer

# From a pre-built matrix
matrix = np.load("data/toy_data/connectivity_matrix.npy")
analyzer = MatrixAnalyzer(matrix=matrix)

# Plot hops starting from specific node indices
analyzer.plot_flow([0, 1, 2], n_steps=3)
```

## Features

- **Flexible input**: accepts edge lists (with configurable column names) or pre-built matrices
- **Weight aggregation**: sum, count, mean, or median for repeated edges
- **Cluster-aware sorting**: rows/columns ordered by cluster then by output strength
- **Top-percent filtering**: carry only the strongest fraction of targets per hop
- **Cluster color strips**: color bars along rows/columns indicate cluster membership
- **Global or per-subplot normalization**: shared or independent color scales
- **Interactive widget**: Jupyter widget with dropdown and slider controls

## Interactive Usage

```python
from connmatrixhops import ConnMatrixHopsWidget

widget = ConnMatrixHopsWidget(analyzer)
widget.interactive()
```

## Project Structure

```
ConnMatrixHops/
├── connmatrixhops/       # Package source
│   ├── __init__.py
│   ├── analyzer.py       # MatrixAnalyzer: core traversal & data logic
│   ├── plotting.py       # Visualization (heatmaps, cluster strips)
│   └── widgets.py        # Jupyter ipywidgets wrapper
├── data/toy_data/        # Example input data
├── figures/              # Example output plots
├── notebooks/            # Demo notebooks
├── tests/                # Unit tests
├── environment/          # Code Ocean environment (Dockerfile, postInstall)
└── run                   # Code Ocean entrypoint script
```

## Code Ocean

This repo is structured as a [Code Ocean](https://codeocean.com/) capsule.

- **Environment**: `environment/Dockerfile` uses a Python 3.11 base image; `environment/postInstall` installs all dependencies via pip.
- **Run**: the `run` script executes both notebooks in sequence and writes outputs to `/results`.
- **Data**: toy data lives in `data/toy_data/` and is accessed via relative paths from the notebooks.

To reproduce locally with Docker:

```bash
docker build -t connmatrixhops -f environment/Dockerfile .
docker run -v $(pwd):/code -v /tmp/results:/results connmatrixhops /code/run
```

## License

MIT
