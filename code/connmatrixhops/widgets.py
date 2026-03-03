"""
Jupyter interactive wrapper for ConnMatrixHops.
"""

import logging
from typing import Optional, Union, List

import ipywidgets as widgets
from IPython.display import display

from connmatrixhops.analyzer import MatrixAnalyzer

logger = logging.getLogger(__name__)


class ConnMatrixHopsWidget:
    """
    Interactive Jupyter widget for exploring N-hop connectivity.

    Allows passing a list of cell IDs or a cluster name to trigger hop analysis.
    Optionally uses ipywidgets for interactive selection.
    """

    def __init__(
        self,
        analyzer: MatrixAnalyzer,
        default_n_hops: int = 4,
        default_top_percent: float = 1.0,
    ):
        """
        Parameters
        ----------
        analyzer : MatrixAnalyzer
            The analyzer instance.
        default_n_hops : int
            Default number of hop subplots.
        default_top_percent : float
            Default fraction (0, 1] of targets to carry forward per hop.
        """
        self.analyzer = analyzer
        self.default_n_hops = default_n_hops
        self.default_top_percent = default_top_percent

    def explore(
        self,
        seed_ids: Optional[Union[List, str]] = None,
        n_hops: Optional[int] = None,
        top_percent: Optional[float] = None,
    ):
        """
        Run hop analysis and display the flow plot.

        Parameters
        ----------
        seed_ids : list of int, or str, optional
            Starting cell IDs or cluster name. If None, uses first cluster or first few cells.
        n_hops : int, optional
            Number of hops. Default from constructor.
        top_percent : float, optional
            Fraction (0, 1] of targets to carry forward. Default from constructor.
        """
        n = n_hops or self.default_n_hops
        tp = top_percent if top_percent is not None else self.default_top_percent

        if seed_ids is None:
            if self.analyzer.metadata is not None and self.analyzer.cluster_col is not None:
                first_cluster = self.analyzer.metadata[self.analyzer.cluster_col].iloc[0]
                seed_ids = first_cluster
                logger.info("Using first cluster as seed: %s", first_cluster)
            else:
                seed_ids = list(self.analyzer._id_to_idx.keys())[:5]
                logger.info("Using first 5 cell IDs as seed: %s", seed_ids)

        return self.analyzer.plot_flow(seed_ids, n_steps=n, top_percent=tp)

    def interactive(
        self,
        cluster_dropdown: bool = True,
        n_hops_slider: bool = True,
        top_percent_slider: bool = True,
    ):
        """
        Create an interactive widget with dropdown/slider controls.

        Parameters
        ----------
        cluster_dropdown : bool
            If True and metadata/cluster_col exist, add cluster selector.
        n_hops_slider : bool
            If True, add slider for n_hops.
        top_percent_slider : bool
            If True, add slider for top_percent filtering.
        """
        out = widgets.Output()

        if cluster_dropdown and self.analyzer.metadata is not None and self.analyzer.cluster_col is not None:
            clusters = sorted(
                self.analyzer.metadata[self.analyzer.cluster_col].unique(),
                key=str,
            )
            cluster_dd = widgets.Dropdown(
                options=clusters,
                value=clusters[0] if clusters else None,
                description="Cluster:",
            )
        else:
            cluster_dd = None

        if n_hops_slider:
            n_slider = widgets.IntSlider(
                value=self.default_n_hops,
                min=1,
                max=8,
                description="Hops:",
            )
        else:
            n_slider = None

        if top_percent_slider:
            tp_slider = widgets.FloatSlider(
                value=self.default_top_percent,
                min=0.01,
                max=1.0,
                step=0.01,
                description="Top %:",
                readout_format=".0%",
            )
        else:
            tp_slider = None

        def on_run(_):
            seed = cluster_dd.value if cluster_dd else list(self.analyzer._id_to_idx.keys())[0]
            n = n_slider.value if n_slider else self.default_n_hops
            tp = tp_slider.value if tp_slider else self.default_top_percent
            with out:
                out.clear_output(wait=True)
                self.analyzer.plot_flow(seed, n_steps=n, top_percent=tp)

        run_btn = widgets.Button(description="Plot")
        run_btn.on_click(on_run)

        items = [i for i in [cluster_dd, n_slider, tp_slider] if i is not None]
        items.append(run_btn)
        box = widgets.HBox(items)
        display(box)
        display(out)
        on_run(None)
