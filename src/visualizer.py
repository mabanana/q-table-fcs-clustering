"""
Visualization Module

Generates plots and reports for science fair presentation.
"""

import logging
import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Creates visualizations for Q-learning clustering results.
    """
    
    def __init__(
        self,
        output_directory: str = "output/plots",
        dpi: int = 300,
        figure_dimensions: tuple = (10, 8),
        color_palette: str = "viridis"
    ):
        """
        Initialize visualization engine.
        
        Args:
            output_directory: Where to save plot files
            dpi: Image resolution
            figure_dimensions: Default figure size (width, height)
            color_palette: Matplotlib/seaborn color scheme
        """
        self.plot_dir = output_directory
        self.resolution = dpi
        self.default_size = figure_dimensions
        self.colors = color_palette
        
        # Create output directory
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = self.resolution
        
        logger.info(f"Visualization engine initialized: {self.plot_dir}")
    
    # Maximum moving-average window when auto-computing
    _MA_WINDOW_MAX = 50
    # Table figure height constants (base + per-row increment)
    _TABLE_FIG_HEIGHT_BASE = 1.5
    _TABLE_FIG_HEIGHT_PER_ROW = 0.6
    _TABLE_FIG_HEIGHT_MIN = 2.0

    def _compute_window_size(self, n_episodes: int, override: int = None) -> int:
        """Return moving-average window size for *n_episodes* episodes."""
        if override is not None:
            return override
        return max(1, min(self._MA_WINDOW_MAX, max(n_episodes // 10, 1)))

    def plot_learning_curve(
        self,
        training_history: List[Dict],
        phase_name: str,
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Plot reward progression over training iterations.
        
        Args:
            training_history: List of iteration statistics
            phase_name: Name of training phase
            save_filename: Optional filename to save plot
            
        Returns:
            Matplotlib Figure object
        """
        if not training_history:
            logger.warning("No training history to plot")
            return None
        
        # Extract data
        iterations = [entry['iteration'] for entry in training_history]
        cumulative_payoffs = [entry['cumulative_payoff'] for entry in training_history]
        average_payoffs = [entry['average_payoff'] for entry in training_history]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.default_size)
        
        # Plot cumulative payoffs
        ax1.plot(iterations, cumulative_payoffs, linewidth=2, color='navy', alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cumulative Payoff')
        ax1.set_title(f'{phase_name} - Cumulative Payoff Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        window_size = min(50, len(cumulative_payoffs) // 10)
        if window_size > 1:
            moving_avg = pd.Series(cumulative_payoffs).rolling(window=window_size).mean()
            ax1.plot(iterations, moving_avg, 'r--', linewidth=2, label=f'{window_size}-iter MA')
            ax1.legend()
        
        # Plot average payoffs
        ax2.plot(iterations, average_payoffs, linewidth=2, color='darkgreen', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Payoff')
        ax2.set_title(f'{phase_name} - Average Payoff Per Sample')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if filename provided
        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved learning curve: {filepath}")
        
        return fig
    
    def plot_q_value_heatmap(
        self,
        q_table: np.ndarray,
        action_labels: List[str],
        n_states: int = None,
        n_state_features: int = None,
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Visualize Q-table as a heatmap.
        
        Args:
            q_table: Q-value matrix (states x actions)
            action_labels: Labels for actions (e.g. ['k=2', 'k=3', ...])
            n_states: Total number of states, used for subtitle
            n_state_features: Number of features used for state encoding
            save_filename: Optional filename to save plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(q_table, cmap=self.colors, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Q-Value', rotation=270, labelpad=20, fontsize=12)
        
        # Set labels
        ax.set_xlabel('Action (Cluster Count k)', fontsize=12)
        ax.set_ylabel('State Index', fontsize=12)
        
        # Build title with optional state-encoding subtitle
        if n_states is not None:
            feat_str = (
                f"{n_state_features}-feature state encoding"
                if n_state_features else "state encoding"
            )
            ax.set_title(
                f'Learned Q-Value Table\n({n_states} states, {feat_str})',
                fontsize=14
            )
        else:
            ax.set_title('Learned Q-Value Table', fontsize=14)
        
        # Set action labels (k=2, k=3, ...) on x-axis
        if len(action_labels) <= 20:
            ax.set_xticks(range(len(action_labels)))
            ax.set_xticklabels(action_labels, fontsize=11)
        
        plt.tight_layout()
        
        # Save if filename provided
        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved Q-value heatmap: {filepath}")
        
        return fig
    
    def plot_action_distribution(
        self,
        action_counts: Dict[int, int],
        action_labels: Dict[int, str],
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Plot distribution of selected actions.
        
        Args:
            action_counts: Dictionary mapping action to count
            action_labels: Dictionary mapping action to label
            save_filename: Optional filename to save plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.default_size)
        
        # Prepare data
        actions = sorted(action_counts.keys())
        counts = [action_counts[a] for a in actions]
        labels = [action_labels.get(a, str(a)) for a in actions]
        
        # Create bar plot
        bars = ax.bar(range(len(actions)), counts, color=sns.color_palette(self.colors, len(actions)))
        
        # Customize
        ax.set_xlabel('Cluster Count (k)', fontsize=12)
        ax.set_ylabel('Frequency Selected', fontsize=12)
        ax.set_title('Action Distribution During Training', fontsize=14)
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(labels)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if filename provided
        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved action distribution: {filepath}")
        
        return fig
    
    def plot_feature_distributions(
        self,
        data: pd.DataFrame,
        bin_boundaries: Dict[str, List[float]],
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Plot feature distributions with discretization boundaries overlaid.
        
        Args:
            data: DataFrame with features
            bin_boundaries: Dictionary mapping feature name to bin edges
            save_filename: Optional filename to save plot
            
        Returns:
            Matplotlib Figure object
        """
        n_features = len(bin_boundaries)
        fig, axes = plt.subplots(n_features, 1, figsize=(self.default_size[0], 4*n_features))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, (feature_name, boundaries) in enumerate(bin_boundaries.items()):
            ax = axes[idx]
            
            # Find matching column
            matching_cols = [col for col in data.columns if feature_name.lower() in col.lower()]
            if not matching_cols:
                continue
            
            col_name = matching_cols[0]
            values = data[col_name].dropna()
            
            # Plot histogram
            ax.hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Overlay bin boundaries
            for boundary in boundaries[1:-1]:  # Skip first and last (0 and inf)
                if boundary < np.inf:
                    ax.axvline(boundary, color='red', linestyle='--', linewidth=2, label='Bin boundary')
            
            ax.set_xlabel(col_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col_name} with Clinical Bins')
            
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend([handles[0]], ['Bin boundary'])
        
        plt.tight_layout()
        
        # Save if filename provided
        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved feature distributions: {filepath}")
        
        return fig
    
    def plot_cluster_scatter(
        self,
        data: pd.DataFrame,
        cluster_labels: np.ndarray,
        feature_x: str,
        feature_y: str,
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Create 2D scatter plot of clustered data.
        
        Args:
            data: DataFrame with features
            cluster_labels: Cluster assignments
            feature_x: Name of x-axis feature
            feature_y: Name of y-axis feature
            save_filename: Optional filename to save plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.default_size)
        
        # Find matching columns
        x_cols = [col for col in data.columns if feature_x.lower() in col.lower()]
        y_cols = [col for col in data.columns if feature_y.lower() in col.lower()]
        
        if not x_cols or not y_cols:
            logger.warning(f"Features {feature_x} or {feature_y} not found")
            return fig
        
        x_data = data[x_cols[0]].values
        y_data = data[y_cols[0]].values
        
        # Create scatter plot
        scatter = ax.scatter(
            x_data,
            y_data,
            c=cluster_labels,
            cmap=self.colors,
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', rotation=270, labelpad=20)
        
        # Labels
        ax.set_xlabel(x_cols[0])
        ax.set_ylabel(y_cols[0])
        ax.set_title(f'Clustering Results: {feature_x} vs {feature_y}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if filename provided
        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved cluster scatter: {filepath}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        class_names: List[str] = None,
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Plot confusion matrix for classification results.
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted labels
            class_names: Names of classes
            save_filename: Optional filename to save plot
            
        Returns:
            Matplotlib Figure object
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot as heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        # Labels
        if class_names:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        # Save if filename provided
        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved confusion matrix: {filepath}")
        
        return fig
    
    def plot_reward_vs_episodes(
        self,
        phase1_history: List[Dict],
        phase2_history: List[Dict],
        phase1_reward_label: str = "Phase 1: Silhouette Score",
        phase2_reward_label: str = "Phase 2: F1 Score",
        window_size: int = None,
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Plot reward vs episodes for both training phases on shared axes.

        Phase 1 (silhouette) and Phase 2 (F1/MCC) are shown in distinct colours
        with a vertical dashed marker at the phase boundary and optional
        moving-average smoothing.

        Args:
            phase1_history: Phase 1 training history records
            phase2_history: Phase 2 training history records
            phase1_reward_label: Legend label for Phase 1 reward
            phase2_reward_label: Legend label for Phase 2 reward
            window_size: Moving-average window (auto-computed if None)
            save_filename: Optional filename to save plot

        Returns:
            Matplotlib Figure object
        """
        if not phase1_history and not phase2_history:
            logger.warning("No training history to plot")
            return None

        fig, ax = plt.subplots(figsize=self.default_size)
        phase1_n = len(phase1_history)

        def _smooth(values, win):
            return pd.Series(values).rolling(window=win, min_periods=1).mean().tolist()

        # Phase 1
        if phase1_history:
            ep1 = [e['iteration'] for e in phase1_history]
            r1 = [e['average_payoff'] for e in phase1_history]
            win = self._compute_window_size(phase1_n, window_size)
            ax.plot(ep1, r1, color='steelblue', alpha=0.3, linewidth=1)
            ax.plot(ep1, _smooth(r1, win), color='steelblue', linewidth=2,
                    label=f'{phase1_reward_label} ({win}-ep MA)')

        # Phase 2 — offset x-axis so episodes are continuous
        if phase2_history:
            phase2_n = len(phase2_history)
            ep2 = [phase1_n + e['iteration'] for e in phase2_history]
            r2 = [e['average_payoff'] for e in phase2_history]
            win = self._compute_window_size(phase2_n, window_size)
            ax.plot(ep2, r2, color='darkorange', alpha=0.3, linewidth=1)
            ax.plot(ep2, _smooth(r2, win), color='darkorange', linewidth=2,
                    label=f'{phase2_reward_label} ({win}-ep MA)')

        # Phase boundary
        if phase1_history and phase2_history:
            ax.axvline(x=phase1_n, color='gray', linestyle='--', linewidth=1.5,
                       label='Phase boundary')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Reward vs Episodes (Both Training Phases)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved reward vs episodes: {filepath}")

        return fig

    def plot_metric_vs_episodes(
        self,
        phase1_history: List[Dict],
        phase2_history: List[Dict],
        metric_key: str = "average_payoff",
        phase1_label: str = "Phase 1",
        phase2_label: str = "Phase 2",
        y_label: str = "Metric Value",
        window_size: int = None,
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Plot a training metric over episodes with one subplot per active phase.

        Args:
            phase1_history: Phase 1 training history (pass [] to skip)
            phase2_history: Phase 2 training history (pass [] to skip)
            metric_key: Key in history dict to plot (e.g. 'average_payoff')
            phase1_label: Legend/title label for Phase 1
            phase2_label: Legend/title label for Phase 2
            y_label: Y-axis label
            window_size: Moving-average window (auto-computed if None)
            save_filename: Optional filename to save plot

        Returns:
            Matplotlib Figure object
        """
        active = [(h, l, c) for h, l, c in [
            (phase1_history, phase1_label, 'steelblue'),
            (phase2_history, phase2_label, 'darkorange'),
        ] if h]

        if not active:
            logger.warning("No training history to plot")
            return None

        n_plots = len(active)
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.default_size[0],
                                                       self.default_size[1]),
                                 squeeze=False)
        axes = axes.flatten()

        for ax, (history, label, color) in zip(axes, active):
            episodes = [e['iteration'] for e in history]
            values = [e[metric_key] for e in history]
            win = self._compute_window_size(len(history), window_size)
            ma = pd.Series(values).rolling(window=win, min_periods=1).mean()

            ax.plot(episodes, values, color=color, alpha=0.3, linewidth=1)
            ax.plot(episodes, ma, color=color, linewidth=2,
                    label=f'{label} ({win}-ep MA)')
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f'{label} — {y_label} Over Training', fontsize=13)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved metric vs episodes: {filepath}")

        return fig

    def plot_marker_scatter_comparison(
        self,
        data: pd.DataFrame,
        baseline_labels: np.ndarray,
        q_labels: np.ndarray,
        feature_x: str = "FS Lin",
        feature_y: str = "SS Log",
        sample_id: str = None,
        baseline_k: int = None,
        q_k: int = None,
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Side-by-side 2D scatter showing baseline-k vs Q-chosen-k clustering.

        Args:
            data: DataFrame containing flow cytometry feature columns
            baseline_labels: Cluster assignments from baseline method
            q_labels: Cluster assignments from Q-table method
            feature_x: X-axis marker name (substring match against columns)
            feature_y: Y-axis marker name (substring match against columns)
            sample_id: Optional sample identifier shown in figure title
            baseline_k: Number of clusters used by baseline (for title)
            q_k: Number of clusters chosen by Q-table (for title)
            save_filename: Optional filename to save plot

        Returns:
            Matplotlib Figure object
        """
        # Resolve column names
        x_cols = [c for c in data.columns if feature_x.lower() in c.lower()]
        y_cols = [c for c in data.columns if feature_y.lower() in c.lower()]

        if not x_cols or not y_cols:
            logger.warning(f"Features '{feature_x}' or '{feature_y}' not found in data")
            fig, _ = plt.subplots(figsize=self.default_size)
            return fig

        x_col, y_col = x_cols[0], y_cols[0]
        x_data = data[x_col].values
        y_data = data[y_col].values

        fig, (ax_base, ax_q) = plt.subplots(1, 2, figsize=(self.default_size[0] * 2,
                                                             self.default_size[1]))

        id_str = f" — {sample_id}" if sample_id else ""

        for ax, labels, k, method in [
            (ax_base, baseline_labels, baseline_k, "Baseline (Fixed k)"),
            (ax_q, q_labels, q_k, "Q-Table"),
        ]:
            k_str = f", k={k}" if k is not None else ""
            sc = ax.scatter(x_data, y_data, c=labels, cmap='tab10',
                            s=40, alpha=0.6, edgecolors='none')
            plt.colorbar(sc, ax=ax, label='Cluster')
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'{method}{k_str}{id_str}', fontsize=13)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f'Clustering Comparison: {x_col} vs {y_col}{id_str}',
            fontsize=14, y=1.02
        )
        plt.tight_layout()

        if save_filename:
            filepath = os.path.join(self.plot_dir, save_filename)
            fig.savefig(filepath, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved scatter comparison: {filepath}")

        return fig

    def save_performance_table(
        self,
        results: Dict,
        output_csv: str = None,
        render_png: bool = True,
        save_filename: Optional[str] = None
    ) -> str:
        """
        Save a performance-summary table comparing baselines vs Q-table.

        Args:
            results: Nested dict of shape
                ``{method_name: {metric_name: value, ...}, ...}``
                e.g. ``{'Fixed k=4': {'Accuracy': 0.70, 'F1': 0.65}, ...}``
            output_csv: Full path for the CSV file; defaults to
                ``<plot_dir>/../performance_summary.csv``
            render_png: When True, also render the table as a PNG figure
            save_filename: PNG filename saved under ``plot_dir``
                (used only when render_png=True)

        Returns:
            Path to the saved CSV file
        """
        if not results:
            logger.warning("Empty results dict — nothing to save")
            return ""

        df = pd.DataFrame(results).T
        df.index.name = "Method"

        # CSV path
        if output_csv is None:
            output_csv = os.path.join(
                os.path.dirname(self.plot_dir.rstrip("/\\")),
                "performance_summary.csv"
            )
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv)
        logger.info(f"Saved performance summary CSV: {output_csv}")

        # PNG table figure
        if render_png:
            fname = save_filename or "performance_summary.png"
            n_rows = len(df)
            fig_h = max(
                self._TABLE_FIG_HEIGHT_MIN,
                n_rows * self._TABLE_FIG_HEIGHT_PER_ROW + self._TABLE_FIG_HEIGHT_BASE
            )
            fig, ax = plt.subplots(figsize=(10, fig_h))
            ax.axis('off')

            col_labels = list(df.columns)
            cell_data = []
            for method, row in df.iterrows():
                cell_row = [str(method)]
                for v in row:
                    cell_row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
                cell_data.append(cell_row)

            tbl = ax.table(
                cellText=cell_data,
                colLabels=["Method"] + col_labels,
                cellLoc='center',
                loc='center'
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.scale(1.2, 1.8)

            ax.set_title(
                'Performance Summary: Baselines vs Q-Table',
                fontsize=13, pad=16
            )
            plt.tight_layout()

            png_path = os.path.join(self.plot_dir, fname)
            fig.savefig(png_path, dpi=self.resolution, bbox_inches='tight')
            logger.info(f"Saved performance summary PNG: {png_path}")
            plt.close(fig)

        return output_csv

    def generate_summary_report(
        self,
        phase1_history: List[Dict],
        phase2_history: List[Dict],
        q_stats: Dict,
        output_filename: str = "training_summary.html"
    ) -> str:
        """
        Generate HTML summary report.
        
        Args:
            phase1_history: Phase 1 training history
            phase2_history: Phase 2 training history
            q_stats: Q-table statistics
            output_filename: HTML filename
            
        Returns:
            Path to generated HTML file
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Q-Learning FCS Clustering - Training Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .metric-value {{ font-size: 1.2em; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>Q-Learning FCS Clustering System</h1>
            <h2>Training Summary Report</h2>
            
            <h3>Phase 1: Quality Learning</h3>
            <div class="metric">
                <span class="metric-label">Iterations:</span>
                <span class="metric-value">{len(phase1_history)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Final Average Payoff:</span>
                <span class="metric-value">{phase1_history[-1]['average_payoff']:.4f}</span>
            </div>
            
            <h3>Phase 2: Diagnostic Refinement</h3>
            <div class="metric">
                <span class="metric-label">Iterations:</span>
                <span class="metric-value">{len(phase2_history)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Final Average Payoff:</span>
                <span class="metric-value">{phase2_history[-1]['average_payoff']:.4f}</span>
            </div>
            
            <h3>Q-Table Statistics</h3>
            <div class="metric">
                <span class="metric-label">State Coverage:</span>
                <span class="metric-value">{q_stats.get('state_coverage', 0):.1f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Mean Q-Value:</span>
                <span class="metric-value">{q_stats.get('mean_q', 0):.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Max Q-Value:</span>
                <span class="metric-value">{q_stats.get('max_q', 0):.4f}</span>
            </div>
        </body>
        </html>
        """
        
        filepath = os.path.join(self.plot_dir, output_filename)
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated summary report: {filepath}")
        return filepath
