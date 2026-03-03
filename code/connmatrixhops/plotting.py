"""
Visualization helpers for ConnMatrixHops.
Separated from analyzer logic for clear data vs. visualization concerns.
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)


def _get_cluster_palette(cluster_labels: list) -> dict:
    """Generate unique color for each unique cluster value."""
    unique = sorted(set(cluster_labels), key=str)
    try:
        cmap = plt.colormaps["tab10"].resampled(max(10, len(unique)))
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10", max(10, len(unique)))
    return {u: cmap(i) for i, u in enumerate(unique)}


def _cluster_vector(
    indices: np.ndarray,
    idx_to_cluster: dict,
) -> np.ndarray:
    """Map matrix indices to cluster indices for color mapping."""
    out = np.full(len(indices), -1)
    for i, idx in enumerate(indices):
        out[i] = idx_to_cluster.get(idx, -1)
    return out


def _cluster_indices_for_strip(cluster_vec: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Convert cluster vector to numeric indices for pcolormesh.
    Returns (numeric_vector, unique_labels_in_order).
    """
    unique = []
    seen = set()
    numeric = np.zeros(len(cluster_vec), dtype=int)
    for i, c in enumerate(cluster_vec):
        if c not in seen:
            seen.add(c)
            unique.append(c)
        numeric[i] = unique.index(c)
    return numeric, unique


def plot_flow(
    analyzer,
    seed_ids,
    n_steps: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    global_norm: bool = True,
    top_percent: float = 1.0,
    axs=None,
):
    """
    Plot N-hop flow subplots with cluster strips.

    Parameters
    ----------
    analyzer : MatrixAnalyzer
        The analyzer instance.
    seed_ids : list, array, or str
        Starting cell IDs or cluster name.
    n_steps : int
        Number of hop subplots.
    figsize : tuple, optional
        Figure size. Default scales with n_steps.
    global_norm : bool
        If True, use shared color scale across subplots. If False, per-subplot.
    top_percent : float
        Fraction (0, 1] of postsynaptic targets to carry forward per hop.
    axs : array of Axes, optional
        If provided, use these axes instead of creating a new figure.
    """
    hops = analyzer.get_hops(seed_ids, n_hops=n_steps, top_percent=top_percent)
    if not hops:
        logger.warning("No hops to plot.")
        return None

    matrix = analyzer._matrix
    if global_norm:
        vmin = float(matrix.min())
        vmax = float(matrix.max())
        if vmin == vmax:
            vmax = vmin + 1e-9
    else:
        vmin, vmax = None, None

    # Build idx -> cluster mapping if metadata exists
    idx_to_cluster = {}
    cluster_palette = {}
    if analyzer.metadata is not None and analyzer.cluster_col is not None:
        for _, row in analyzer.metadata.iterrows():
            cid = row[analyzer.cell_id_col]
            clab = row[analyzer.cluster_col]
            if cid in analyzer._id_to_idx:
                idx = analyzer._id_to_idx[cid]
                idx_to_cluster[idx] = clab
        all_clusters = list(set(idx_to_cluster.values()))
        cluster_palette = _get_cluster_palette(np.array(all_clusters))

    n_plots = len(hops)
    if figsize is None:
        figsize = (6 * n_plots, 5)
    if axs is None:
        fig, axs = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)
        axs = axs[0]
    else:
        fig = axs[0].figure

    for k, (row_idx, col_idx) in enumerate(hops):
        ax = axs[k] if k < len(axs) else axs[-1]

        if len(col_idx) == 0:
            ax.text(0.5, 0.5, "No downstream targets", ha="center", va="center")
            ax.set_title(f"Hop {k}")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sub = matrix[np.ix_(row_idx, col_idx)]

        smin, smax = (vmin, vmax) if global_norm else (None, None)
        if not global_norm and sub.size > 0:
            smin, smax = float(sub.min()), float(sub.max())
            if smin == smax:
                smax = smin + 1e-9

        # Main heatmap
        im = ax.imshow(
            sub,
            aspect="auto",
            cmap="viridis",
            vmin=smin,
            vmax=smax,
            interpolation="nearest",
        )

        # Title: show filtering info when active
        if top_percent < 1.0 and k > 0:
            pct_label = f"{top_percent * 100:g}%"
            title = f"Hop {k}  (top {pct_label}, {len(row_idx)} rows)"
        else:
            title = f"Hop {k}  ({len(row_idx)} rows)"
        ax.set_title(title)

        # Cluster strips on left and top
        has_clusters = bool(idx_to_cluster)
        if has_clusters:
            row_clusters = [idx_to_cluster.get(i, "") for i in row_idx]
            col_clusters = [idx_to_cluster.get(i, "") for i in col_idx]

            row_num, row_unique = _cluster_indices_for_strip(np.array(row_clusters))
            col_num, col_unique = _cluster_indices_for_strip(np.array(col_clusters))

            all_unique = list(dict.fromkeys(list(row_unique) + list(col_unique)))
            n_clusters = max(len(all_unique), 1)
            row_num = np.array([all_unique.index(c) for c in row_clusters])
            col_num = np.array([all_unique.index(c) for c in col_clusters])
            colors = [cluster_palette.get(u, (0.8, 0.8, 0.8, 1)) for u in all_unique]
            if not colors:
                colors = [(0.8, 0.8, 0.8, 1)]
            cmap_cluster = mcolors.ListedColormap(colors)

            strip_width = 0.04
            strip_height = 0.04

            # Left strip
            left_ax = ax.inset_axes([-strip_width - 0.02, 0, strip_width, 1], transform=ax.transAxes)
            row_mat = row_num.reshape(-1, 1)
            left_ax.imshow(row_mat, aspect="auto", cmap=cmap_cluster, vmin=0, vmax=n_clusters - 1)
            left_ax.set_xticks([])
            left_ax.set_yticks([])

            # Top strip
            top_ax = ax.inset_axes([0, 1.02, 1, strip_height], transform=ax.transAxes)
            col_mat = col_num.reshape(1, -1)
            top_ax.imshow(col_mat, aspect="auto", cmap=cmap_cluster, vmin=0, vmax=n_clusters - 1)
            top_ax.set_xticks([])
            top_ax.set_yticks([])

        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    if global_norm and n_plots > 0:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Connectivity strength")
    else:
        fig.tight_layout()

    return fig
