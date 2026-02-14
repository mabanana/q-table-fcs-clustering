"""
Feature Discretizer Module

Implements discretization of continuous flow cytometry data into discrete bins
based on feature-specific thresholds.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalDiscretizer:
    """
    Discretizes continuous flow cytometry data into feature-based bins.
    """

    DEFAULT_FEATURE_BINS = [0, 0.5, 1.0, 2.0, 3.0, np.inf]

    def __init__(
        self,
        feature_bins: Optional[Dict[str, List[float]]] = None,
        state_features: Optional[List[str]] = None
    ):
        """
        Initialize the discretizer with feature-specific bin boundaries.

        Args:
            feature_bins: Mapping of feature name to bin edges
            state_features: Feature names used for state encoding
        """
        self.feature_bins = feature_bins or {}
        self.state_features = state_features or []

        logger.info("Initialized ClinicalDiscretizer with:")
        logger.info(f"  State features: {self.state_features}")
        if self.feature_bins:
            logger.info(f"  Feature bins: {self.feature_bins}")

    def _get_bins(self, feature_name: str) -> List[float]:
        return self.feature_bins.get(feature_name, self.DEFAULT_FEATURE_BINS)

    def discretize_feature(self, values: np.ndarray, bins: List[float]) -> np.ndarray:
        """
        Discretize values into bins.

        Args:
            values: Array of feature values
            bins: Bin edges

        Returns:
            Array of bin indices (0-3)
        """
        thresholds = bins[1:-1] if len(bins) > 2 else bins
        bin_indices = np.digitize(values, thresholds, right=False)
        max_bin = max(len(bins) - 2, 0)
        return np.clip(bin_indices, 0, max_bin)
    
    def discretize_data(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Discretize selected features in a DataFrame.

        Args:
            data: DataFrame containing flow cytometry data
            feature_names: Features to discretize (defaults to configured state features)

        Returns:
            DataFrame with discretized feature bins added
        """
        result = data.copy()
        features = feature_names or self.state_features

        if not features:
            raise ValueError("No feature names provided for discretization")

        for feature in features:
            if feature in result.columns:
                bins = self._get_bins(feature)
                result[f"{feature}_bin"] = self.discretize_feature(
                    result[feature].values,
                    bins
                )
                logger.debug(f"Discretized {feature} into {feature}_bin")
            else:
                logger.warning(f"Column {feature} not found in data")

        return result
    
    def create_state(self, bin_indices: List[int], bin_sizes: Optional[List[int]] = None) -> int:
        """
        Create a discrete state representation from bin indices.

        Uses mixed-radix encoding across features.
        """
        if not bin_indices:
            raise ValueError("bin_indices must not be empty")

        if bin_sizes is None:
            bin_sizes = [4] * len(bin_indices)

        if len(bin_sizes) != len(bin_indices):
            raise ValueError("bin_sizes length must match bin_indices length")

        state = 0
        for bin_index, base in zip(bin_indices, bin_sizes):
            base = max(int(base), 1)
            state = state * base + int(bin_index)

        return state
    
    def decode_state(
        self,
        state: int,
        num_features: int = 2,
        bin_sizes: Optional[List[int]] = None
    ) -> Tuple[int, ...]:
        """
        Decode a state representation back into bin indices.

        Args:
            state: Integer state representation
            num_features: Number of features

        Returns:
            Tuple of bin indices
        """
        if bin_sizes is None:
            bin_sizes = [4] * num_features

        bins = []
        remainder = state
        for base in reversed(bin_sizes):
            base = max(int(base), 1)
            bins.append(remainder % base)
            remainder //= base
        return tuple(reversed(bins))
    
    def get_state_from_data(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert discretized data to state representations.

        Args:
            data: DataFrame with discretized feature bins
            feature_names: Features to encode (defaults to configured state features)

        Returns:
            Array of state representations
        """
        features = feature_names or self.state_features

        if not features:
            raise ValueError("No feature names provided for state encoding")

        bin_sizes = [max(len(self._get_bins(feature)) - 1, 1) for feature in features]
        states = []

        for _, row in data.iterrows():
            bin_indices = [int(row.get(f"{feature}_bin", 0)) for feature in features]
            bin_indices = [min(max(idx, 0), size - 1) for idx, size in zip(bin_indices, bin_sizes)]
            state = self.create_state(bin_indices, bin_sizes=bin_sizes)
            states.append(state)

        return np.array(states)
    
    def get_clinical_interpretation(
        self,
        bin_indices: List[int],
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        Get interpretation for a given bin combination.

        Args:
            bin_indices: Bin indices per feature
            feature_names: Feature names (defaults to configured state features)

        Returns:
            Human-readable interpretation
        """
        features = feature_names or self.state_features

        interpretation_parts = []
        for feature, bin_index in zip(features, bin_indices):
            bins = self._get_bins(feature)
            if 0 <= bin_index < len(bins) - 1:
                bin_range = f"[{bins[bin_index]}, {bins[bin_index + 1]})"
                interpretation_parts.append(f"{feature}: {bin_range}")

        return " | ".join(interpretation_parts)
    
    def get_bin_statistics(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Get statistics about bin distributions in the data.

        Args:
            data: DataFrame with discretized features
            feature_names: Features to include (defaults to configured state features)

        Returns:
            Dictionary with statistics
        """
        stats = {}
        features = feature_names or self.state_features

        for feature in features:
            column = f"{feature}_bin"
            if column in data.columns:
                stats[f"{feature}_bin_counts"] = data[column].value_counts().to_dict()
                stats[f"{feature}_bin_distribution"] = (
                    data[column].value_counts(normalize=True).to_dict()
                )

        return stats
    
    def explain_discretization(self) -> str:
        """
        Generate a detailed explanation of the discretization scheme.
        
        Returns:
            Formatted string explaining the clinical basis for bins
        """
        explanation = []
        explanation.append("=" * 80)
        explanation.append("FEATURE DISCRETIZATION SCHEME")
        explanation.append("=" * 80)
        explanation.append("")

        features = self.state_features or list(self.feature_bins.keys())
        if not features:
            explanation.append("No features configured for discretization")
            explanation.append("=" * 80)
            return "\n".join(explanation)

        for feature in features:
            bins = self._get_bins(feature)
            explanation.append(f"{feature} BINS:")
            explanation.append("-" * 80)
            for i in range(len(bins) - 1):
                bin_range = f"[{bins[i]}, {bins[i+1]})"
                explanation.append(f"  Bin {i}: {bin_range:20s}")
            explanation.append("")

        explanation.append("=" * 80)

        return "\n".join(explanation)
