"""
Smoke tests for the VisualizationEngine module.

These tests verify that all plotting functions execute without errors and
produce the expected output files.  They do NOT validate visual quality.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.visualizer import VisualizationEngine


def _make_history(n=20, base_payoff=0.3, slope=0.001):
    """Return a minimal fake training-history list."""
    return [
        {
            'iteration': i,
            'cumulative_payoff': i * base_payoff,
            'average_payoff': base_payoff + i * slope,
            'exploration_rate': max(0.05, 0.3 - i * 0.001),
            'action_diversity': 3,
        }
        for i in range(n)
    ]


class TestVisualizationEngine:
    """Smoke tests for VisualizationEngine plotting utilities."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.viz = VisualizationEngine(output_directory=self.tmpdir, dpi=72)
        self.phase1_history = _make_history(20, 0.3, 0.001)
        self.phase2_history = _make_history(20, 0.5, 0.002)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization_creates_output_dir(self):
        assert os.path.isdir(self.tmpdir)
        assert self.viz.plot_dir == self.tmpdir

    # ------------------------------------------------------------------
    # Existing methods (regression smoke tests)
    # ------------------------------------------------------------------

    def test_plot_learning_curve_saves_file(self):
        fig = self.viz.plot_learning_curve(
            self.phase1_history, "Phase 1", save_filename="lc_smoke.png"
        )
        assert fig is not None
        assert os.path.isfile(os.path.join(self.tmpdir, "lc_smoke.png"))

    def test_plot_action_distribution_saves_file(self):
        action_counts = {i: 10 + i * 5 for i in range(9)}
        action_labels = {i: f"k={i + 2}" for i in range(9)}
        fig = self.viz.plot_action_distribution(
            action_counts, action_labels, save_filename="act_dist_smoke.png"
        )
        assert fig is not None
        assert os.path.isfile(os.path.join(self.tmpdir, "act_dist_smoke.png"))

    def test_plot_confusion_matrix_saves_file(self):
        true_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pred_labels = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        fig = self.viz.plot_confusion_matrix(
            true_labels, pred_labels,
            class_names=["negative", "positive"],
            save_filename="cm_smoke.png",
        )
        assert fig is not None
        assert os.path.isfile(os.path.join(self.tmpdir, "cm_smoke.png"))

    # ------------------------------------------------------------------
    # Improved plot_q_value_heatmap
    # ------------------------------------------------------------------

    def test_plot_q_value_heatmap_with_n_states(self):
        q_table = np.random.rand(16, 9)
        action_labels = [f"k={k}" for k in range(2, 11)]
        fig = self.viz.plot_q_value_heatmap(
            q_table, action_labels,
            n_states=16, n_state_features=2,
            save_filename="heatmap_smoke.png",
        )
        assert fig is not None
        assert os.path.isfile(os.path.join(self.tmpdir, "heatmap_smoke.png"))

    def test_plot_q_value_heatmap_without_n_states(self):
        """Backward-compatible call without n_states."""
        q_table = np.random.rand(16, 9)
        action_labels = [f"k={k}" for k in range(2, 11)]
        fig = self.viz.plot_q_value_heatmap(q_table, action_labels)
        assert fig is not None

    # ------------------------------------------------------------------
    # plot_reward_vs_episodes
    # ------------------------------------------------------------------

    def test_plot_reward_vs_episodes_both_phases_saves_file(self):
        fig = self.viz.plot_reward_vs_episodes(
            self.phase1_history,
            self.phase2_history,
            save_filename="reward_ep_smoke.png",
        )
        assert fig is not None
        assert os.path.isfile(os.path.join(self.tmpdir, "reward_ep_smoke.png"))

    def test_plot_reward_vs_episodes_phase1_only(self):
        fig = self.viz.plot_reward_vs_episodes(self.phase1_history, [])
        assert fig is not None

    def test_plot_reward_vs_episodes_phase2_only(self):
        fig = self.viz.plot_reward_vs_episodes([], self.phase2_history)
        assert fig is not None

    def test_plot_reward_vs_episodes_empty_returns_none(self):
        fig = self.viz.plot_reward_vs_episodes([], [])
        assert fig is None

    # ------------------------------------------------------------------
    # plot_metric_vs_episodes
    # ------------------------------------------------------------------

    def test_plot_metric_vs_episodes_both_phases_saves_file(self):
        fig = self.viz.plot_metric_vs_episodes(
            self.phase1_history,
            self.phase2_history,
            save_filename="metric_ep_smoke.png",
        )
        assert fig is not None
        assert os.path.isfile(os.path.join(self.tmpdir, "metric_ep_smoke.png"))

    def test_plot_metric_vs_episodes_phase2_only(self):
        fig = self.viz.plot_metric_vs_episodes(
            [],
            self.phase2_history,
            metric_key="average_payoff",
            phase2_label="Phase 2: F1 Score",
            y_label="F1 Score",
        )
        assert fig is not None

    def test_plot_metric_vs_episodes_empty_returns_none(self):
        fig = self.viz.plot_metric_vs_episodes([], [])
        assert fig is None

    # ------------------------------------------------------------------
    # plot_marker_scatter_comparison
    # ------------------------------------------------------------------

    def test_plot_marker_scatter_comparison_saves_file(self):
        n = 50
        data = pd.DataFrame({
            "FS Lin": np.random.rand(n) * 4,
            "SS Log": np.random.rand(n) * 3,
        })
        baseline_labels = np.random.randint(0, 4, n)
        q_labels = np.random.randint(0, 6, n)

        fig = self.viz.plot_marker_scatter_comparison(
            data,
            baseline_labels,
            q_labels,
            feature_x="FS Lin",
            feature_y="SS Log",
            sample_id="patient_001",
            baseline_k=4,
            q_k=6,
            save_filename="scatter_smoke.png",
        )
        assert fig is not None
        assert os.path.isfile(os.path.join(self.tmpdir, "scatter_smoke.png"))

    def test_plot_marker_scatter_comparison_missing_features(self):
        """Returns a figure (empty) when columns are not found."""
        data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        fig = self.viz.plot_marker_scatter_comparison(
            data,
            np.array([0, 1]),
            np.array([0, 1]),
            feature_x="FS Lin",
            feature_y="SS Log",
        )
        assert fig is not None

    # ------------------------------------------------------------------
    # save_performance_table
    # ------------------------------------------------------------------

    def test_save_performance_table_csv(self):
        results = {
            "Fixed k=4": {"Accuracy": 0.70, "F1": 0.65, "MCC": 0.55, "Mean k": 4.0},
            "Silhouette k": {"Accuracy": 0.73, "F1": 0.68, "MCC": 0.58, "Mean k": 5.2},
            "Q-Table": {"Accuracy": 0.80, "F1": 0.76, "MCC": 0.66, "Mean k": 4.8},
        }
        csv_path = os.path.join(self.tmpdir, "perf_smoke.csv")
        returned = self.viz.save_performance_table(
            results, output_csv=csv_path, render_png=False
        )
        assert returned == csv_path
        assert os.path.isfile(csv_path)
        df = pd.read_csv(csv_path, index_col=0)
        assert list(df.index) == ["Fixed k=4", "Silhouette k", "Q-Table"]

    def test_save_performance_table_png(self):
        results = {
            "Fixed k=4": {"Accuracy": 0.70, "F1": 0.65},
            "Q-Table": {"Accuracy": 0.80, "F1": 0.76},
        }
        self.viz.save_performance_table(
            results, render_png=True, save_filename="perf_table_smoke.png"
        )
        assert os.path.isfile(os.path.join(self.tmpdir, "perf_table_smoke.png"))

    def test_save_performance_table_empty_returns_empty_string(self):
        result = self.viz.save_performance_table({})
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
