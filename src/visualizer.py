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
        save_filename: Optional[str] = None
    ) -> Figure:
        """
        Visualize Q-table as a heatmap.
        
        Args:
            q_table: Q-value matrix (states x actions)
            action_labels: Labels for actions
            save_filename: Optional filename to save plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(q_table, cmap=self.colors, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Q-Value', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_xlabel('Action (Cluster Count)')
        ax.set_ylabel('State')
        ax.set_title('Learned Q-Value Table')
        
        # Set action labels
        if len(action_labels) <= 20:  # Only show labels if not too many
            ax.set_xticks(range(len(action_labels)))
            ax.set_xticklabels(action_labels)
        
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
        ax.set_xlabel('Cluster Count')
        ax.set_ylabel('Frequency Selected')
        ax.set_title('Action Distribution During Training')
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
