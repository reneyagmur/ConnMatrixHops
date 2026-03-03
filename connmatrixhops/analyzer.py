"""
MatrixAnalyzer — Core logic for N-hop connectivity traversal and visualization.
"""

import logging
from typing import Optional, Union, List, Tuple, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WeightAggFunc = Literal["sum", "count", "mean", "median"]


class MatrixAnalyzer:
    """
    Analyzes directed graph connectivity and supports N-hop traversal visualization.
    Data-agnostic: works with any edge list and metadata column names.
    """

    def __init__(
        self,
        edge_list: Optional[pd.DataFrame] = None,
        matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        metadata: Optional[pd.DataFrame] = None,
        source_col: str = "pre_id",
        target_col: str = "post_id",
        weight_col: Optional[str] = None,
        weight_agg_func: Optional[WeightAggFunc] = None,
        cell_id_col: str = "cell_id",
        cluster_col: Optional[str] = None,
    ):
        """
        Initialize the analyzer from either an edge list or a pre-built matrix.

        Parameters
        ----------
        edge_list : pd.DataFrame, optional
            Edge list with source/target columns.
        matrix : np.ndarray or pd.DataFrame, optional
            Pre-built C×C connectivity matrix. Must provide either edge_list or matrix.
        metadata : pd.DataFrame, optional
            Cell metadata with cell IDs and cluster labels.
        source_col : str
            Column name for source/pre-synaptic IDs in edge list. Default 'pre_id'.
        target_col : str
            Column name for target/post-synaptic IDs in edge list. Default 'post_id'.
        weight_col : str, optional
            Column name for edge weights. If None, uses count. Default None.
        weight_agg_func : str, optional
            Aggregation for repeated edges: 'sum', 'count', 'mean', 'median'.
            Default: 'count' if weight_col is None, else 'sum'.
        cell_id_col : str
            Column name for cell IDs in metadata. Default 'cell_id'.
        cluster_col : str, optional
            Column name for cluster labels in metadata. Enables cluster-based sorting.
        """
        if edge_list is None and matrix is None:
            raise ValueError("Must provide either edge_list or matrix.")
        if edge_list is not None and matrix is not None:
            raise ValueError("Provide only one of edge_list or matrix.")

        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.weight_agg_func = weight_agg_func or ("count" if weight_col is None else "sum")
        self.cell_id_col = cell_id_col
        self.cluster_col = cluster_col
        self.metadata = metadata

        if matrix is not None:
            self._matrix, self._id_to_idx, self._idx_to_id = self._from_matrix(matrix)
        else:
            self._matrix, self._id_to_idx, self._idx_to_id = self._build_matrix_from_edges(
                edge_list
            )

        self.n_cells = len(self._id_to_idx)
        self._sorted_indices: Optional[np.ndarray] = None
        if metadata is not None and cluster_col is not None:
            self._sorted_indices = self._compute_sorted_indices()

    def _from_matrix(
        self, matrix: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, dict, dict]:
        """Build internal state from a pre-existing matrix."""
        arr = np.asarray(matrix)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Matrix must be square (C×C).")
        n = arr.shape[0]
        id_to_idx = {i: i for i in range(n)}
        idx_to_id = {i: i for i in range(n)}

        # If metadata provided, ensure all metadata IDs are in the index
        if self.metadata is not None:
            ids = self.metadata[self.cell_id_col].unique()
            if not all(i in id_to_idx for i in ids):
                logger.warning(
                    "Some metadata IDs are not in matrix indices (0..n-1). "
                    "Re-indexing may be required."
                )
        return arr.astype(float), id_to_idx, idx_to_id

    def _build_matrix_from_edges(self, edge_list: pd.DataFrame) -> Tuple[np.ndarray, dict, dict]:
        """Build dense connectivity matrix from edge list with optional aggregation."""
        src = edge_list[self.source_col].values
        tgt = edge_list[self.target_col].values

        # Collect all cell IDs
        all_ids = set(np.unique(np.concatenate([src, tgt])))
        if self.metadata is not None:
            meta_ids = set(self.metadata[self.cell_id_col].dropna().unique())
            all_ids = all_ids | meta_ids

        all_ids = sorted(all_ids)
        id_to_idx = {uid: i for i, uid in enumerate(all_ids)}
        idx_to_id = {i: uid for uid, i in id_to_idx.items()}
        n = len(all_ids)

        matrix = np.zeros((n, n))

        if self.weight_col is not None and self.weight_col in edge_list.columns:
            agg_df = edge_list.groupby([self.source_col, self.target_col])[
                self.weight_col
            ].agg(self.weight_agg_func)
            for (s, t), w in agg_df.items():
                if s in id_to_idx and t in id_to_idx:
                    matrix[id_to_idx[s], id_to_idx[t]] = w
        else:
            agg_df = edge_list.groupby([self.source_col, self.target_col]).size()
            for (s, t), count in agg_df.items():
                if s in id_to_idx and t in id_to_idx:
                    matrix[id_to_idx[s], id_to_idx[t]] = count

        logger.info("Built %d×%d connectivity matrix from %d edges.", n, n, len(edge_list))
        return matrix, id_to_idx, idx_to_id

    def _compute_sorted_indices(self) -> np.ndarray:
        """
        Sort indices: first by cluster (via cluster_col), then within cluster by total output.
        """
        if self.metadata is None or self.cluster_col is None:
            return np.arange(self.n_cells)

        meta = self.metadata.copy()
        meta = meta[meta[self.cell_id_col].isin(self._id_to_idx)]
        meta["_idx"] = meta[self.cell_id_col].map(self._id_to_idx)
        meta = meta.dropna(subset=["_idx"])
        meta["_idx"] = meta["_idx"].astype(int)

        row_sums = self._matrix.sum(axis=1)
        meta["_row_sum"] = meta["_idx"].map(lambda i: row_sums[i])

        # Sort by cluster, then by row sum descending within cluster
        meta = meta.sort_values(
            [self.cluster_col, "_row_sum"], ascending=[True, False]
        )
        return meta["_idx"].values

    def _resolve_ids(self, ids: Union[List, str]) -> np.ndarray:
        """Convert cell IDs or cluster name to matrix indices."""
        if isinstance(ids, str):
            if self.metadata is None or self.cluster_col is None:
                raise ValueError("Cluster name lookup requires metadata and cluster_col.")
            mask = self.metadata[self.cluster_col] == ids
            ids = self.metadata.loc[mask, self.cell_id_col].tolist()
        ids = np.atleast_1d(ids)
        indices = []
        for uid in ids:
            if uid in self._id_to_idx:
                indices.append(self._id_to_idx[uid])
            else:
                logger.warning("Cell ID %s not found in matrix, skipping.", uid)
        return np.array(indices, dtype=int) if indices else np.array([], dtype=int)

    def get_hops(
        self,
        start_nodes: Union[List, np.ndarray, str],
        n_hops: int = 4,
        top_percent: float = 1.0,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        BFS-style traversal: for each hop, return (row_indices, col_indices) of the submatrix.

        Hop 0: start_nodes as rows, their targets as columns.
        Hop 1: targets of hop 0 as rows, their targets as columns, etc.

        When top_percent < 1.0, only the top fraction of postsynaptic targets
        (ranked by total incoming connection strength from the current hop)
        propagate as rows into the next hop.

        Parameters
        ----------
        start_nodes : list, array, or str
            Starting cell IDs or cluster name.
        n_hops : int
            Number of hops.
        top_percent : float
            Fraction (0, 1] of postsynaptic targets to carry forward per hop,
            ranked by column-sum of incoming strength. Default 1.0 (all targets).

        Returns
        -------
        List of (row_idx, col_idx) arrays, one per hop.
        """
        if not 0 < top_percent <= 1.0:
            raise ValueError("top_percent must be in (0, 1].")

        seed_indices = self._resolve_ids(start_nodes)
        if len(seed_indices) == 0:
            logger.warning("No valid seed nodes; returning empty hop list.")
            return []

        result: List[Tuple[np.ndarray, np.ndarray]] = []
        current_rows = seed_indices

        for _ in range(n_hops):
            row_sub = self._matrix[current_rows, :]
            has_targets = row_sub.any(axis=0)
            col_indices = np.where(has_targets)[0]

            result.append((current_rows.copy(), col_indices))

            if len(col_indices) == 0:
                logger.info("Hop reached zero downstream targets; stopping early.")
                break

            # Filter to top_percent of targets by incoming strength
            if top_percent < 1.0 and len(col_indices) > 1:
                col_strengths = row_sub[:, col_indices].sum(axis=0)
                n_keep = max(1, int(np.ceil(len(col_indices) * top_percent)))
                top_k = np.argsort(col_strengths)[-n_keep:]
                col_indices_filtered = col_indices[np.sort(top_k)]
                current_rows = np.unique(col_indices_filtered)
            else:
                current_rows = np.unique(col_indices)

        return result

    def get_hop_indices(
        self,
        seed_ids: Union[List, np.ndarray, str],
        n_steps: int = 4,
        top_percent: float = 1.0,
    ) -> List[np.ndarray]:
        """
        BFS-style traversal: return list of index arrays (one per hop).

        Hop 0: seed_ids.
        Hop 1: unique targets of Hop 0.
        Hop 2: unique targets of Hop 1, etc.
        """
        hops = self.get_hops(seed_ids, n_hops=n_steps, top_percent=top_percent)
        return [h[0] for h in hops]  # row indices per hop

    @property
    def matrix(self) -> np.ndarray:
        """Full connectivity matrix (optionally with sorted row/col order)."""
        if self._sorted_indices is not None:
            idx = self._sorted_indices
            return self._matrix[np.ix_(idx, idx)]
        return self._matrix

    @property
    def sorted_indices(self) -> Optional[np.ndarray]:
        """Indices for cluster-sorted matrix view, or None."""
        return self._sorted_indices

    def get_matrix_for_display(self) -> np.ndarray:
        """Matrix in display order (sorted if metadata/cluster_col provided)."""
        return self.matrix

    def plot_flow(
        self,
        seed_ids: Union[List, np.ndarray, str],
        n_steps: int = 4,
        figsize: Optional[Tuple[float, float]] = None,
        global_norm: bool = True,
        top_percent: float = 1.0,
    ):
        """
        Plot N-hop flow subplots with cluster strips.

        Parameters
        ----------
        seed_ids : list, array, or str
            Starting cell IDs or cluster name.
        n_steps : int
            Number of hop subplots.
        figsize : tuple, optional
            Figure size.
        global_norm : bool
            If True, shared color scale; if False, per-subplot scaling.
        top_percent : float
            Fraction (0, 1] of postsynaptic targets to carry forward per hop.
        """
        from connmatrixhops.plotting import plot_flow as _plot_flow

        return _plot_flow(
            self, seed_ids, n_steps=n_steps, figsize=figsize,
            global_norm=global_norm, top_percent=top_percent,
        )

    def plot_hops(
        self,
        start_nodes: Union[List, np.ndarray, str],
        n_hops: int = 4,
        figsize: Optional[Tuple[float, float]] = None,
        global_norm: bool = True,
        top_percent: float = 1.0,
    ):
        """Alias for plot_flow."""
        return self.plot_flow(
            start_nodes, n_steps=n_hops, figsize=figsize,
            global_norm=global_norm, top_percent=top_percent,
        )
