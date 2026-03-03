# ConnMatrixHops — Unified Specification

**Role:** Expert Python Data Scientist, Neuroinformatics Developer & Software Architect for Scientific Computing

---

## Context

ConnMatrixHops is a tool to visualize signal flow through N-hop subcircuits in a connectivity matrix. The core logic involves selecting "Source" neurons (seeds), finding their postsynaptic targets, and treating those targets as the "Source" for the next subplot, effectively tracing a path through the network.

**Design Philosophy:** Build a flexible, data-agnostic system. Do NOT hard-code specific cluster names, column names, or data values. The class should handle any directed graph data and its associated metadata.
- Package Structure: Ensure the code is structured for a package (e.g., proper __init__.py in the connmatrixhops/ directory).

---

## 1. Data Input & Processing

### Input Type A (Matrix)

- A **C×C** NumPy array or Pandas DataFrame representing the connectivity matrix directly.

### Input Type B (Edge List)

- A Pandas/Polars DataFrame with source and target columns.
- **User-Defined Column Mapping:** Allow users to pass:
  - `source_col` (default: `'pre_id'`)
  - `target_col` (default: `'post_id'`)
  - `weight_col` (optional, e.g. `'synapse_size'`)
- **Weight Aggregation:** Support a `weight_agg_func` parameter when multiple edges exist between the same pair:
  - Options: `'sum'`, `'count'`, `'mean'`, `'median'`
  - Default: `'count'` if no weight column, otherwise `'sum'`
- **Matrix Consistency:** Build an internal dense matrix. If a metadata table is provided, ensure all IDs in the metadata are represented in the matrix, even if they have no synapses.

### Cluster Metadata

- A DataFrame with cell IDs and cluster labels.
- **Flexible Metadata:** Accept an optional DataFrame for cell metadata and a `cluster_col` string to identify which column defines the groups.
- **Dynamic Sorting (optional):** First by cluster (via `cluster_col`), then within each cluster by "total output connectivity strength" (row sum). Must work regardless of how many clusters exist or what they are named.

---

## 2. The ConnMatrixHops / MatrixAnalyzer Class

Create a class `MatrixAnalyzer` in `analyzer.py` that:

- Stores the global connectivity matrix and metadata.
- Keeps clear separation between **Data Processing** and **Visualization** methods.

### Method: `get_hops(start_nodes, n_hops=4)` / `get_hop_indices(seed_ids, n_steps=4)`

- **BFS-style traversal** over the directed graph:
  - Hop 0: `seed_ids` (the starting neurons).
  - Hop 1: Unique postsynaptic targets of Hop 0.
  - Hop 2: Unique postsynaptic targets of Hop 1.
  - … and so on for `n_hops` / `n_steps`.
- Returns a list of index arrays, one per hop, suitable for subplot rows.

### Method: `plot_hops()` / `plot_flow(seed_ids, n_steps=4)`

- Uses **Matplotlib** (or optionally Plotly) to generate the N-subplot layout.
- **Subplot Layout:** One subplot per hop, showing the relevant submatrix (rows = "source" neurons for that hop, columns = their postsynaptic targets).
- **Normalization:**
  - Default: **Global** — shared color scale across all subplots (compute min/max from full connectivity matrix).
  - Optional: **Per-subplot** (sub-global) color scaling.
- **Visuals:**
  - Use `'viridis'` colormap for heatmaps.
  - If metadata is present, draw 1D color bars/strips along the **top** and **left** of the main matrix to represent clusters. Auto-generate a unique color palette for whatever categories exist in `cluster_col`.
- **Labels:** Hide individual cell IDs (too dense); label cluster blocks only.
- **Empty State:** If a hop results in zero downstream targets, display "No downstream targets" instead of crashing.

---

## 3. Interactivity (Wrapper)

- Provide a function or class for **Jupyter** that allows:
  - Passing a list of cell IDs or a cluster name to trigger the hop analysis and visualization.
  - **(Bonus)** Use `ipywidgets` or Plotly events to allow clicking a row or a cluster color bar to update the subplots interactively.

---

## 4. Technical Specifications

- **Stack:** `matplotlib`, `seaborn`, `numpy`, `pandas`. 
- **Modular layout:** `analyzer.py` for core logic; a demo notebook for usage.
- 

---

## 5. Code Quality

- **No hard-coded strings** for data values (cluster names, column names in logic).
- Use **logging** instead of `print` for diagnostics and debugging.
- Clear **separation** between Data Processing and Visualization methods.

---

## Summary of Key Parameters

| Parameter        | Purpose                                       | Default                    |
|-----------------|-----------------------------------------------|----------------------------|
| `source_col`    | Edge list source column                       | `'pre_id'`                 |
| `target_col`    | Edge list target column                       | `'post_id'`                |
| `weight_col`    | Edge list weight column (optional)            | `None`                     |
| `weight_agg_func` | Aggregation for repeated edges             | `'count'` / `'sum'`        |
| `cluster_col`   | Metadata column for cluster labels            | User-specified             |
| `n_hops` / `n_steps` | Number of hop subplots                    | `4`                        |
