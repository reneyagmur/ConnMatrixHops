"""Tests for the MatrixAnalyzer class."""

import numpy as np
import pandas as pd
import pytest

from connmatrixhops import MatrixAnalyzer


@pytest.fixture
def simple_matrix():
    """5x5 matrix with a clear chain: 0->1->2->3->4."""
    m = np.zeros((5, 5))
    m[0, 1] = 1.0
    m[1, 2] = 1.0
    m[2, 3] = 1.0
    m[3, 4] = 1.0
    return m


@pytest.fixture
def edge_list_df():
    return pd.DataFrame(
        {
            "pre_id": [10, 20, 30, 40],
            "post_id": [20, 30, 40, 50],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def metadata_df():
    return pd.DataFrame(
        {
            "cell_id": [10, 20, 30, 40, 50],
            "cluster": ["A", "A", "B", "B", "C"],
        }
    )


class TestMatrixInit:
    def test_from_matrix(self, simple_matrix):
        analyzer = MatrixAnalyzer(matrix=simple_matrix)
        assert analyzer.n_cells == 5
        np.testing.assert_array_equal(analyzer._matrix, simple_matrix)

    def test_from_edge_list(self, edge_list_df):
        analyzer = MatrixAnalyzer(
            edge_list=edge_list_df,
            source_col="pre_id",
            target_col="post_id",
            weight_col="weight",
        )
        assert analyzer.n_cells == 5
        assert analyzer._matrix[analyzer._id_to_idx[10], analyzer._id_to_idx[20]] == 1.0

    def test_must_provide_input(self):
        with pytest.raises(ValueError, match="Must provide"):
            MatrixAnalyzer()

    def test_cannot_provide_both(self, simple_matrix, edge_list_df):
        with pytest.raises(ValueError, match="only one"):
            MatrixAnalyzer(matrix=simple_matrix, edge_list=edge_list_df)

    def test_non_square_matrix_rejected(self):
        with pytest.raises(ValueError, match="square"):
            MatrixAnalyzer(matrix=np.zeros((3, 4)))


class TestHopTraversal:
    def test_chain_hops(self, simple_matrix):
        analyzer = MatrixAnalyzer(matrix=simple_matrix)
        hops = analyzer.get_hops([0], n_hops=4)
        assert len(hops) == 4
        np.testing.assert_array_equal(hops[0][0], [0])
        np.testing.assert_array_equal(hops[0][1], [1])
        np.testing.assert_array_equal(hops[1][0], [1])
        np.testing.assert_array_equal(hops[1][1], [2])

    def test_hop_indices(self, simple_matrix):
        analyzer = MatrixAnalyzer(matrix=simple_matrix)
        indices = analyzer.get_hop_indices([0], n_steps=3)
        assert len(indices) == 3
        np.testing.assert_array_equal(indices[0], [0])

    def test_empty_seed_returns_empty(self, simple_matrix):
        analyzer = MatrixAnalyzer(matrix=simple_matrix)
        hops = analyzer.get_hops([], n_hops=2)
        assert hops == []

    def test_top_percent_filtering(self):
        m = np.zeros((4, 4))
        m[0, 1] = 10.0
        m[0, 2] = 1.0
        m[0, 3] = 1.0
        analyzer = MatrixAnalyzer(matrix=m)
        hops = analyzer.get_hops([0], n_hops=2, top_percent=0.5)
        assert len(hops) >= 1


class TestClusterMetadata:
    def test_sorted_indices_computed(self, edge_list_df, metadata_df):
        analyzer = MatrixAnalyzer(
            edge_list=edge_list_df,
            source_col="pre_id",
            target_col="post_id",
            weight_col="weight",
            metadata=metadata_df,
            cell_id_col="cell_id",
            cluster_col="cluster",
        )
        assert analyzer._sorted_indices is not None
        assert len(analyzer._sorted_indices) == len(metadata_df)

    def test_resolve_cluster_name(self, edge_list_df, metadata_df):
        analyzer = MatrixAnalyzer(
            edge_list=edge_list_df,
            source_col="pre_id",
            target_col="post_id",
            metadata=metadata_df,
            cell_id_col="cell_id",
            cluster_col="cluster",
        )
        hops = analyzer.get_hops("A", n_hops=2)
        assert len(hops) > 0


class TestPlotFlow:
    def test_plot_returns_figure(self, simple_matrix):
        import matplotlib
        matplotlib.use("Agg")

        analyzer = MatrixAnalyzer(matrix=simple_matrix)
        fig = analyzer.plot_flow([0], n_steps=2)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")
