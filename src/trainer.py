"""
Trainer Module

Implements a progressive two-phase learning system for reinforcement-based clustering.
Phase 1: Quality-driven learning on homogeneous samples
Phase 2: Diagnosis-driven refinement on heterogeneous samples
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from .fcs_loader import FCSLoader
from .discretizer import ClinicalDiscretizer
from .q_learning import QLearningAgent, create_action_space
from .clustering import ClusteringEngine, assign_diagnosis_from_clusters

logger = logging.getLogger(__name__)


class ReinforcementClusteringPipeline:
    """
    Progressive two-phase reinforcement learning system for medical clustering:
    Phase 1: Quality optimization on positive-only samples
    Phase 2: Diagnostic accuracy refinement on mixed labeled samples
    """
    
    def __init__(
        self,
        q_agent: QLearningAgent,
        action_space: Dict[int, Union[int, Dict[str, Union[int, str]]]],
        discretizer: ClinicalDiscretizer,
        fcs_loader: FCSLoader,
        clustering_engine: ClusteringEngine,
        state_features: Optional[List[str]] = None,
        feature_markers: Optional[List[str]] = None
    ):
        """
        Initialize the reinforcement clustering pipeline.
        
        Args:
            q_agent: Q-learning decision maker
            action_space: Action-to-cluster mapping
            discretizer: Feature discretization module
            fcs_loader: Flow cytometry file loader
            clustering_engine: Clustering computation module
            state_features: Feature names used for state encoding
            feature_markers: Optional list of marker names to use for clustering
        """
        self.decision_maker = q_agent
        self.cluster_action_map = action_space
        self.feature_binner = discretizer
        self.data_reader = fcs_loader
        self.cluster_computer = clustering_engine
        self.state_features = state_features or []
        self.feature_markers = feature_markers
        
        self.phase1_progress = []
        self.phase2_progress = []
        
        logger.info("Initialized ReinforcementClusteringPipeline")
    
    def execute_quality_learning_phase(
        self,
        positive_samples_path: str,
        learning_iterations: int = 1000,
        samples_per_iteration: int = 10,
        quality_measure: str = "silhouette"
    ) -> List[Dict]:
        """
        Phase 1: Quality-based reinforcement learning on homogeneous positive samples.
        
        Objective: Optimize cluster count selection for maximum cluster quality.
        
        Args:
            positive_samples_path: Directory with HIV+ FCS files
            learning_iterations: Number of RL iterations
            samples_per_iteration: Batch size for each iteration
            quality_measure: Quality metric ('silhouette' or 'inertia')
            
        Returns:
            Progress tracking records
        """
        logger.info(f"Phase 1 Quality Learning: {learning_iterations} iterations")
        logger.info(f"Source: {positive_samples_path}")
        logger.info(f"Quality measure: {quality_measure}")
        
        # Load flow cytometry data files
        fcs_file_data = self.data_reader.load_directory(positive_samples_path)
        
        if not fcs_file_data:
            raise ValueError(f"No files in {positive_samples_path}")
        
        logger.info(f"Loaded {len(fcs_file_data)} files")
        
        # Compute summary statistics per file
        summary_stats = self.data_reader.aggregate_data(fcs_file_data, aggregation="median")
        
        # Apply clinical binning
        binned_features = self.feature_binner.discretize_data(
            summary_stats,
            feature_names=self.state_features
        )
        
        # Encode as discrete states
        encoded_states = self.feature_binner.get_state_from_data(
            binned_features,
            feature_names=self.state_features
        )
        
        logger.info(f"State space coverage: {len(np.unique(encoded_states))} unique states")
        
        # Extract clustering input features
        cluster_inputs = self._build_feature_matrix(summary_stats)
        
        # Reinforcement learning loop
        iteration_logs = []
        seed_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
        progress_bar = tqdm(total=learning_iterations, desc="Phase 1")
        for iteration_num in range(learning_iterations):
            iteration_payoffs = []
            selected_actions = []
            
            # Sample mini-batch
            if len(encoded_states) > samples_per_iteration:
                sample_idx = np.random.choice(len(encoded_states), samples_per_iteration, replace=False)
            else:
                sample_idx = np.arange(len(encoded_states))
            
            for idx in sample_idx:
                current_state = encoded_states[idx]
                
                # Decision: select cluster count
                chosen_action = self.decision_maker.choose_action(current_state, explore=True)
                cluster_count, seeding_strategy = self._resolve_action(chosen_action)
                selected_actions.append(chosen_action)
                
                # Execute clustering
                try:
                    cluster_labels, within_variance, _ = self.cluster_computer.kmeans(
                        cluster_inputs,
                        n_clusters=cluster_count,
                        random_state=42,
                        init_centers=self._get_predicted_seed_centers(
                            cluster_inputs,
                            encoded_states,
                            current_state,
                            cluster_count,
                            seed_cache,
                            strategy=seeding_strategy
                        )
                    )
                    
                    # Compute quality-based payoff
                    if quality_measure == "silhouette":
                        payoff = self.cluster_computer.silhouette_score(cluster_inputs, cluster_labels)
                    elif quality_measure == "inertia":
                        payoff = -within_variance / 10000.0  # Normalized negative inertia
                    else:
                        raise ValueError(f"Unknown quality measure: {quality_measure}")
                    
                    iteration_payoffs.append(payoff)
                    
                    # Learn from outcome
                    self.decision_maker.update_q_value(current_state, chosen_action, payoff, current_state)
                    
                except Exception as e:
                    logger.warning(f"Clustering error at k={cluster_count}: {str(e)}")
                    iteration_payoffs.append(-1.0)
                    self.decision_maker.update_q_value(current_state, chosen_action, -1.0, current_state)
            
            # Adapt exploration rate
            self.decision_maker.decay_epsilon()
            
            # Track iteration
            cumulative_payoff = np.sum(iteration_payoffs)
            self.decision_maker.record_episode(cumulative_payoff, selected_actions)
            
            log_entry = {
                'iteration': iteration_num,
                'cumulative_payoff': cumulative_payoff,
                'average_payoff': np.mean(iteration_payoffs),
                'exploration_rate': self.decision_maker.epsilon,
                'action_diversity': len(set(selected_actions))
            }
            iteration_logs.append(log_entry)
            
            # Display progress
            progress_bar.update(1)
            if iteration_num % 100 == 0:
                progress_bar.set_postfix({
                    'payoff': f"{cumulative_payoff:.3f}",
                    'epsilon': f"{self.decision_maker.epsilon:.3f}"
                })
        
        progress_bar.close()
        self.phase1_progress = iteration_logs
        logger.info(f"Phase 1 complete. Exploration rate: {self.decision_maker.epsilon:.4f}")
        
        return iteration_logs
    
    def execute_diagnostic_refinement_phase(
        self,
        labeled_samples_path: str,
        learning_iterations: int = 1000,
        samples_per_iteration: int = 10,
        accuracy_measure: str = "f1"
    ) -> List[Dict]:
        """
        Phase 2: Diagnostic accuracy refinement on labeled heterogeneous samples.
        
        Objective: Fine-tune cluster selection for optimal HIV diagnosis.
        
        Args:
            labeled_samples_path: Directory with labeled FCS files
            learning_iterations: Number of RL iterations
            samples_per_iteration: Batch size
            accuracy_measure: Diagnostic metric ('f1', 'mcc', 'accuracy')
            
        Returns:
            Progress tracking records
        """
        logger.info(f"Phase 2 Diagnostic Refinement: {learning_iterations} iterations")
        logger.info(f"Source: {labeled_samples_path}")
        logger.info(f"Accuracy measure: {accuracy_measure}")
        
        # Load flow cytometry files
        fcs_file_data = self.data_reader.load_directory(labeled_samples_path)
        
        if not fcs_file_data:
            raise ValueError(f"No files in {labeled_samples_path}")
        
        logger.info(f"Loaded {len(fcs_file_data)} files")
        
        # Compute summary statistics
        summary_stats = self.data_reader.aggregate_data(fcs_file_data, aggregation="median")
        
        # Parse diagnosis labels from filenames
        diagnosis_labels = []
        for fname in summary_stats['filename']:
            label = self.data_reader.extract_label_from_filename(fname)
            diagnosis_labels.append(label if label else "unlabeled")
        
        summary_stats['diagnosis'] = diagnosis_labels
        
        # Keep only labeled samples
        labeled_subset = summary_stats[summary_stats['diagnosis'] != "unlabeled"].copy()
        
        if len(labeled_subset) == 0:
            raise ValueError("No labeled samples. Filenames need 'positive' or 'negative'.")
        
        positive_count = np.sum(labeled_subset['diagnosis'] == 'positive')
        negative_count = np.sum(labeled_subset['diagnosis'] == 'negative')
        logger.info(f"Labeled samples: {len(labeled_subset)} (pos={positive_count}, neg={negative_count})")
        
        # Apply clinical binning
        binned_features = self.feature_binner.discretize_data(
            labeled_subset,
            feature_names=self.state_features
        )
        
        # Encode states
        encoded_states = self.feature_binner.get_state_from_data(
            binned_features,
            feature_names=self.state_features
        )
        
        # Extract features and ground truth
        cluster_inputs = self._build_feature_matrix(labeled_subset)
        ground_truth = labeled_subset['diagnosis'].values
        
        # Reinforcement learning loop
        iteration_logs = []
        seed_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
        progress_bar = tqdm(total=learning_iterations, desc="Phase 2")
        for iteration_num in range(learning_iterations):
            iteration_payoffs = []
            selected_actions = []
            
            # Mini-batch sampling
            if len(encoded_states) > samples_per_iteration:
                sample_idx = np.random.choice(len(encoded_states), samples_per_iteration, replace=False)
            else:
                sample_idx = np.arange(len(encoded_states))
            
            for idx in sample_idx:
                current_state = encoded_states[idx]
                
                # Decision: select cluster count
                chosen_action = self.decision_maker.choose_action(current_state, explore=True)
                cluster_count, seeding_strategy = self._resolve_action(chosen_action)
                selected_actions.append(chosen_action)
                
                # Execute clustering
                try:
                    cluster_labels, within_variance, _ = self.cluster_computer.kmeans(
                        cluster_inputs,
                        n_clusters=cluster_count,
                        random_state=42,
                        init_centers=self._get_predicted_seed_centers(
                            cluster_inputs,
                            encoded_states,
                            current_state,
                            cluster_count,
                            seed_cache,
                            strategy=seeding_strategy
                        )
                    )
                    
                    # Map clusters to diagnoses via majority voting
                    cluster_diagnosis_map, predicted_diagnoses = assign_diagnosis_from_clusters(
                        cluster_labels,
                        ground_truth,
                        positive_class="positive"
                    )
                    
                    # Evaluate diagnostic accuracy
                    if accuracy_measure == "f1":
                        payoff = f1_score(
                            ground_truth,
                            predicted_diagnoses,
                            pos_label="positive",
                            average='binary'
                        )
                    elif accuracy_measure == "mcc":
                        # Matthews correlation coefficient normalized to [0,1]
                        binary_truth = (ground_truth == "positive").astype(int)
                        binary_pred = (predicted_diagnoses == "positive").astype(int)
                        mcc_score = matthews_corrcoef(binary_truth, binary_pred)
                        payoff = (mcc_score + 1) / 2
                    elif accuracy_measure == "accuracy":
                        payoff = accuracy_score(ground_truth, predicted_diagnoses)
                    else:
                        raise ValueError(f"Unknown accuracy measure: {accuracy_measure}")
                    
                    iteration_payoffs.append(payoff)
                    
                    # Learn from outcome
                    self.decision_maker.update_q_value(current_state, chosen_action, payoff, current_state)
                    
                except Exception as e:
                    logger.warning(f"Clustering error at k={cluster_count}: {str(e)}")
                    iteration_payoffs.append(0.0)
                    self.decision_maker.update_q_value(current_state, chosen_action, 0.0, current_state)
            
            # Adapt exploration rate
            self.decision_maker.decay_epsilon()
            
            # Track iteration
            cumulative_payoff = np.sum(iteration_payoffs)
            self.decision_maker.record_episode(cumulative_payoff, selected_actions)
            
            log_entry = {
                'iteration': iteration_num,
                'cumulative_payoff': cumulative_payoff,
                'average_payoff': np.mean(iteration_payoffs),
                'exploration_rate': self.decision_maker.epsilon,
                'action_diversity': len(set(selected_actions))
            }
            iteration_logs.append(log_entry)
            
            # Display progress
            progress_bar.update(1)
            if iteration_num % 100 == 0:
                progress_bar.set_postfix({
                    'payoff': f"{cumulative_payoff:.3f}",
                    'epsilon': f"{self.decision_maker.epsilon:.3f}"
                })
        
        progress_bar.close()
        self.phase2_progress = iteration_logs
        logger.info(f"Phase 2 complete. Exploration rate: {self.decision_maker.epsilon:.4f}")
        
        return iteration_logs
    
    def apply_learned_policy(
        self,
        test_samples_path: str,
        predictions_output: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply learned clustering policy to new unlabeled samples.
        
        Args:
            test_samples_path: Directory with unlabeled FCS files
            predictions_output: Optional CSV path for saving results
            
        Returns:
            DataFrame with diagnostic predictions
        """
        logger.info(f"Applying policy to: {test_samples_path}")
        
        # Load test files
        fcs_file_data = self.data_reader.load_directory(test_samples_path)
        
        if not fcs_file_data:
            raise ValueError(f"No files in {test_samples_path}")
        
        logger.info(f"Loaded {len(fcs_file_data)} test files")
        
        # Compute summary statistics
        summary_stats = self.data_reader.aggregate_data(fcs_file_data, aggregation="median")
        
        # Apply clinical binning
        binned_features = self.feature_binner.discretize_data(
            summary_stats,
            feature_names=self.state_features
        )
        
        # Encode states
        encoded_states = self.feature_binner.get_state_from_data(
            binned_features,
            feature_names=self.state_features
        )
        
        # Extract features
        cluster_inputs = self._build_feature_matrix(summary_stats)
        
        # Generate predictions using learned policy
        prediction_records = []
        seed_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
        for sample_idx, current_state in enumerate(encoded_states):
            # Use learned policy (no exploration)
            chosen_action = self.decision_maker.choose_action(current_state, explore=False)
            cluster_count, seeding_strategy = self._resolve_action(chosen_action)
            
            # Cluster all samples
            cluster_labels, _, _ = self.cluster_computer.kmeans(
                cluster_inputs,
                n_clusters=cluster_count,
                random_state=42,
                init_centers=self._get_predicted_seed_centers(
                    cluster_inputs,
                    encoded_states,
                    current_state,
                    cluster_count,
                    seed_cache,
                    strategy=seeding_strategy
                )
            )
            
            # Get this sample's cluster assignment
            assigned_cluster = cluster_labels[sample_idx]
            
            # Simple heuristic: cluster 0 as positive (refineable with more data)
            predicted_status = "positive" if assigned_cluster == 0 else "negative"
            
            prediction_records.append({
                'filename': summary_stats.iloc[sample_idx]['filename'],
                'encoded_state': current_state,
                'selected_cluster_count': cluster_count,
                'selected_seeding_strategy': seeding_strategy,
                'assigned_cluster': assigned_cluster,
                'predicted_hiv_status': predicted_status
            })
        
        results_dataframe = pd.DataFrame(prediction_records)
        
        # Save predictions if path provided
        if predictions_output:
            os.makedirs(os.path.dirname(predictions_output), exist_ok=True)
            results_dataframe.to_csv(predictions_output, index=False)
            logger.info(f"Predictions saved: {predictions_output}")
        
        logger.info(f"Completed predictions for {len(results_dataframe)} samples")
        
        return results_dataframe
    
    def _locate_marker_column(self, dataframe: pd.DataFrame, marker_name: str) -> str:
        """Locate column matching marker name."""
        for col_name in dataframe.columns:
            if marker_name.lower() in col_name.lower():
                return col_name
        return marker_name
    
    def _build_feature_matrix(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Build numerical feature matrix for clustering."""
        feature_columns = []

        if self.feature_markers:
            for marker in self.feature_markers:
                col_name = self._locate_marker_column(dataframe, marker)
                if col_name in dataframe.columns:
                    feature_columns.append(col_name)
        else:
            numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [
                col for col in numeric_columns
                if col not in {"filename", "label", "diagnosis"}
            ]
            feature_columns = numeric_columns

        if not feature_columns:
            numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [
                col for col in numeric_columns
                if col not in {"filename", "label", "diagnosis"}
            ]
            feature_columns = numeric_columns

        if not feature_columns:
            raise ValueError(
                "No feature columns located. "
                f"Available columns: {list(dataframe.columns)}"
            )

        feature_df = dataframe[feature_columns].copy()
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        all_nan_columns = feature_df.columns[feature_df.isna().all()].tolist()
        if all_nan_columns:
            logger.warning(
                "Dropping all-NaN feature columns: %s",
                all_nan_columns
            )
            feature_df = feature_df.drop(columns=all_nan_columns)

        if feature_df.isna().any().any():
            column_medians = feature_df.median()
            feature_df = feature_df.fillna(column_medians)
            if feature_df.isna().any().any():
                feature_df = feature_df.fillna(0.0)

        if feature_df.shape[1] == 0:
            raise ValueError(
                "No usable feature columns after cleaning. "
                f"Available columns: {list(dataframe.columns)}"
            )

        return feature_df.values

    def _get_predicted_seed_centers(
        self,
        cluster_inputs: np.ndarray,
        encoded_states: np.ndarray,
        current_state: int,
        n_clusters: int,
        seed_cache: Dict[Tuple[int, int, str], np.ndarray],
        strategy: str = "predicted_state"
    ) -> np.ndarray:
        """Return deterministic predicted initial centers for K-means."""
        cache_key = (int(current_state), int(n_clusters), strategy)
        if cache_key in seed_cache:
            return seed_cache[cache_key]

        if strategy == "predicted_global":
            state_points = cluster_inputs
        else:
            state_mask = encoded_states == current_state
            state_points = cluster_inputs[state_mask]

            if state_points.shape[0] < n_clusters:
                state_points = cluster_inputs

        centers = self._farthest_point_centers(state_points, n_clusters)
        seed_cache[cache_key] = centers
        return centers

    def _resolve_action(self, chosen_action: int) -> Tuple[int, str]:
        """Resolve action index into (cluster_count, seeding_strategy)."""
        action_value = self.cluster_action_map[chosen_action]
        if isinstance(action_value, dict):
            cluster_count = int(action_value.get("cluster_count", 2))
            seeding_strategy = str(action_value.get("seeding_strategy", "predicted_state"))
            return cluster_count, seeding_strategy

        return int(action_value), "predicted_state"

    def _farthest_point_centers(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """Deterministic center prediction via mean anchor + farthest-point expansion."""
        if data.shape[0] < n_clusters:
            raise ValueError(
                f"Cannot create {n_clusters} centers from only {data.shape[0]} samples"
            )

        selected_indices: List[int] = []

        # First center: point closest to global mean (deterministic)
        mean_point = np.mean(data, axis=0)
        first_idx = int(np.argmin(np.linalg.norm(data - mean_point, axis=1)))
        selected_indices.append(first_idx)

        # Next centers: farthest-point traversal
        while len(selected_indices) < n_clusters:
            selected_points = data[selected_indices]
            distances = np.linalg.norm(
                data[:, np.newaxis, :] - selected_points[np.newaxis, :, :],
                axis=2
            )
            min_distances = np.min(distances, axis=1)

            # Avoid selecting the same index again
            min_distances[selected_indices] = -np.inf
            next_idx = int(np.argmax(min_distances))

            if min_distances[next_idx] == -np.inf:
                # Degenerate case: duplicate points everywhere
                for fallback_idx in range(data.shape[0]):
                    if fallback_idx not in selected_indices:
                        next_idx = fallback_idx
                        break

            selected_indices.append(next_idx)

        return data[selected_indices]
