"""
Clinically-Informed Discretizer Module

Implements discretization of continuous flow cytometry data into discrete bins
based on clinically relevant thresholds (WHO HIV staging criteria, etc.).
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalDiscretizer:
    """
    Discretizes continuous flow cytometry data into clinically-informed bins.
    
    Based on:
    - WHO HIV staging criteria for CD4 counts
    - Clinical significance of CD4/CD8 ratios
    - Activation marker thresholds
    """
    
    # Default bin boundaries based on clinical criteria
    DEFAULT_CD4_BINS = [0, 200, 350, 500, np.inf]
    DEFAULT_CD4_CD8_RATIO_BINS = [0, 1.0, 1.5, 2.5, np.inf]
    DEFAULT_ACTIVATION_BINS = [0, 500, 1500, 3000, np.inf]
    
    # Clinical interpretations for each bin
    CD4_INTERPRETATIONS = [
        "Severe immunodeficiency (<200)",
        "Advanced immunodeficiency (200-350)",
        "Mild immunodeficiency (350-500)",
        "Normal (>500)"
    ]
    
    CD4_CD8_RATIO_INTERPRETATIONS = [
        "Inverted - immunocompromised (<1.0)",
        "Low (1.0-1.5)",
        "Normal (1.5-2.5)",
        "High (>2.5)"
    ]
    
    ACTIVATION_INTERPRETATIONS = [
        "Low activation (<500)",
        "Moderate activation (500-1500)",
        "High activation (1500-3000)",
        "Very high activation (>3000)"
    ]
    
    def __init__(
        self,
        cd4_bins: Optional[List[float]] = None,
        cd4_cd8_ratio_bins: Optional[List[float]] = None,
        activation_bins: Optional[List[float]] = None
    ):
        """
        Initialize the discretizer with custom or default bin boundaries.
        
        Args:
            cd4_bins: Bin boundaries for CD4 counts (default: WHO criteria)
            cd4_cd8_ratio_bins: Bin boundaries for CD4/CD8 ratio
            activation_bins: Bin boundaries for activation markers (e.g., HLA-DR)
        """
        self.cd4_bins = cd4_bins or self.DEFAULT_CD4_BINS
        self.cd4_cd8_ratio_bins = cd4_cd8_ratio_bins or self.DEFAULT_CD4_CD8_RATIO_BINS
        self.activation_bins = activation_bins or self.DEFAULT_ACTIVATION_BINS
        
        logger.info("Initialized ClinicalDiscretizer with:")
        logger.info(f"  CD4 bins: {self.cd4_bins}")
        logger.info(f"  CD4/CD8 ratio bins: {self.cd4_cd8_ratio_bins}")
        logger.info(f"  Activation bins: {self.activation_bins}")
    
    def discretize_cd4(self, cd4_values: np.ndarray) -> np.ndarray:
        """
        Discretize CD4 counts into bins based on WHO HIV staging criteria.
        
        Bins:
        - 0: <200 (Severe immunodeficiency)
        - 1: 200-350 (Advanced immunodeficiency)
        - 2: 350-500 (Mild immunodeficiency)
        - 3: >500 (Normal)
        
        Args:
            cd4_values: Array of CD4 count values
            
        Returns:
            Array of bin indices (0-3)
        """
        return np.digitize(cd4_values, self.cd4_bins[1:], right=False)
    
    def discretize_cd4_cd8_ratio(self, ratio_values: np.ndarray) -> np.ndarray:
        """
        Discretize CD4/CD8 ratio into bins based on clinical significance.
        
        Bins:
        - 0: <1.0 (Inverted - immunocompromised)
        - 1: 1.0-1.5 (Low)
        - 2: 1.5-2.5 (Normal)
        - 3: >2.5 (High)
        
        Args:
            ratio_values: Array of CD4/CD8 ratio values
            
        Returns:
            Array of bin indices (0-3)
        """
        return np.digitize(ratio_values, self.cd4_cd8_ratio_bins[1:], right=False)
    
    def discretize_activation(self, activation_values: np.ndarray) -> np.ndarray:
        """
        Discretize activation marker values (e.g., HLA-DR) into bins.
        
        Bins:
        - 0: <500 (Low)
        - 1: 500-1500 (Moderate)
        - 2: 1500-3000 (High)
        - 3: >3000 (Very high)
        
        Args:
            activation_values: Array of activation marker values
            
        Returns:
            Array of bin indices (0-3)
        """
        return np.digitize(activation_values, self.activation_bins[1:], right=False)
    
    def discretize_data(
        self,
        data: pd.DataFrame,
        cd4_column: str = "CD4",
        cd8_column: str = "CD8",
        activation_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Discretize all relevant features in a DataFrame.
        
        Args:
            data: DataFrame containing flow cytometry data
            cd4_column: Name of CD4 column
            cd8_column: Name of CD8 column
            activation_column: Optional name of activation marker column (e.g., HLA-DR)
            
        Returns:
            DataFrame with discretized features added
        """
        result = data.copy()
        
        # Discretize CD4
        if cd4_column in result.columns:
            result['cd4_bin'] = self.discretize_cd4(result[cd4_column].values)
            logger.debug(f"Discretized {cd4_column} into cd4_bin")
        else:
            logger.warning(f"Column {cd4_column} not found in data")
        
        # Calculate and discretize CD4/CD8 ratio
        if cd4_column in result.columns and cd8_column in result.columns:
            # Avoid division by zero
            cd8_safe = result[cd8_column].replace(0, 1e-10)
            result['cd4_cd8_ratio'] = result[cd4_column] / cd8_safe
            result['cd4_cd8_ratio_bin'] = self.discretize_cd4_cd8_ratio(
                result['cd4_cd8_ratio'].values
            )
            logger.debug(f"Calculated and discretized CD4/CD8 ratio")
        else:
            logger.warning(f"Columns {cd4_column} or {cd8_column} not found in data")
        
        # Discretize activation marker if available
        if activation_column and activation_column in result.columns:
            result['activation_bin'] = self.discretize_activation(
                result[activation_column].values
            )
            logger.debug(f"Discretized {activation_column} into activation_bin")
        
        return result
    
    def create_state(
        self,
        cd4_bin: int,
        cd4_cd8_ratio_bin: int,
        activation_bin: Optional[int] = None
    ) -> int:
        """
        Create a discrete state representation from bin indices.
        
        State encoding:
        - 2 features (CD4, CD4/CD8 ratio): state = cd4_bin * 4 + cd4_cd8_ratio_bin
        - 3 features (+ activation): state = cd4_bin * 16 + cd4_cd8_ratio_bin * 4 + activation_bin
        
        Args:
            cd4_bin: CD4 bin index (0-3)
            cd4_cd8_ratio_bin: CD4/CD8 ratio bin index (0-3)
            activation_bin: Optional activation marker bin index (0-3)
            
        Returns:
            Integer state representation
        """
        if activation_bin is not None:
            # 3 features: 4^3 = 64 possible states
            state = cd4_bin * 16 + cd4_cd8_ratio_bin * 4 + activation_bin
        else:
            # 2 features: 4^2 = 16 possible states
            state = cd4_bin * 4 + cd4_cd8_ratio_bin
        
        return state
    
    def decode_state(
        self,
        state: int,
        num_features: int = 2
    ) -> Tuple[int, int, Optional[int]]:
        """
        Decode a state representation back into bin indices.
        
        Args:
            state: Integer state representation
            num_features: Number of features (2 or 3)
            
        Returns:
            Tuple of (cd4_bin, cd4_cd8_ratio_bin, activation_bin)
            activation_bin is None for 2-feature encoding
        """
        if num_features == 3:
            cd4_bin = state // 16
            remainder = state % 16
            cd4_cd8_ratio_bin = remainder // 4
            activation_bin = remainder % 4
            return cd4_bin, cd4_cd8_ratio_bin, activation_bin
        else:
            cd4_bin = state // 4
            cd4_cd8_ratio_bin = state % 4
            return cd4_bin, cd4_cd8_ratio_bin, None
    
    def get_state_from_data(
        self,
        data: pd.DataFrame,
        use_activation: bool = False
    ) -> np.ndarray:
        """
        Convert discretized data to state representations.
        
        Args:
            data: DataFrame with discretized features (cd4_bin, cd4_cd8_ratio_bin, etc.)
            use_activation: Whether to include activation marker in state
            
        Returns:
            Array of state representations
        """
        states = []
        
        for idx, row in data.iterrows():
            cd4_bin = int(row.get('cd4_bin', 0))
            cd4_cd8_ratio_bin = int(row.get('cd4_cd8_ratio_bin', 0))
            
            if use_activation and 'activation_bin' in row:
                activation_bin = int(row['activation_bin'])
            else:
                activation_bin = None
            
            state = self.create_state(cd4_bin, cd4_cd8_ratio_bin, activation_bin)
            states.append(state)
        
        return np.array(states)
    
    def get_clinical_interpretation(
        self,
        cd4_bin: int,
        cd4_cd8_ratio_bin: int,
        activation_bin: Optional[int] = None
    ) -> str:
        """
        Get clinical interpretation for a given bin combination.
        
        Args:
            cd4_bin: CD4 bin index
            cd4_cd8_ratio_bin: CD4/CD8 ratio bin index
            activation_bin: Optional activation marker bin index
            
        Returns:
            Human-readable clinical interpretation
        """
        interpretation_parts = []
        
        if 0 <= cd4_bin < len(self.CD4_INTERPRETATIONS):
            interpretation_parts.append(f"CD4: {self.CD4_INTERPRETATIONS[cd4_bin]}")
        
        if 0 <= cd4_cd8_ratio_bin < len(self.CD4_CD8_RATIO_INTERPRETATIONS):
            interpretation_parts.append(
                f"CD4/CD8 ratio: {self.CD4_CD8_RATIO_INTERPRETATIONS[cd4_cd8_ratio_bin]}"
            )
        
        if activation_bin is not None and 0 <= activation_bin < len(self.ACTIVATION_INTERPRETATIONS):
            interpretation_parts.append(
                f"Activation: {self.ACTIVATION_INTERPRETATIONS[activation_bin]}"
            )
        
        return " | ".join(interpretation_parts)
    
    def get_bin_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Get statistics about bin distributions in the data.
        
        Args:
            data: DataFrame with discretized features
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if 'cd4_bin' in data.columns:
            stats['cd4_bin_counts'] = data['cd4_bin'].value_counts().to_dict()
            stats['cd4_bin_distribution'] = (
                data['cd4_bin'].value_counts(normalize=True).to_dict()
            )
        
        if 'cd4_cd8_ratio_bin' in data.columns:
            stats['cd4_cd8_ratio_bin_counts'] = (
                data['cd4_cd8_ratio_bin'].value_counts().to_dict()
            )
            stats['cd4_cd8_ratio_bin_distribution'] = (
                data['cd4_cd8_ratio_bin'].value_counts(normalize=True).to_dict()
            )
        
        if 'activation_bin' in data.columns:
            stats['activation_bin_counts'] = (
                data['activation_bin'].value_counts().to_dict()
            )
            stats['activation_bin_distribution'] = (
                data['activation_bin'].value_counts(normalize=True).to_dict()
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
        explanation.append("CLINICALLY-INFORMED DISCRETIZATION SCHEME")
        explanation.append("=" * 80)
        explanation.append("")
        
        explanation.append("CD4 COUNT BINS (WHO HIV Staging Criteria):")
        explanation.append("-" * 80)
        for i, interp in enumerate(self.CD4_INTERPRETATIONS):
            bin_range = f"[{self.cd4_bins[i]}, {self.cd4_bins[i+1]})"
            explanation.append(f"  Bin {i}: {bin_range:20s} - {interp}")
        explanation.append("")
        
        explanation.append("CD4/CD8 RATIO BINS (Clinical Significance):")
        explanation.append("-" * 80)
        for i, interp in enumerate(self.CD4_CD8_RATIO_INTERPRETATIONS):
            bin_range = f"[{self.cd4_cd8_ratio_bins[i]}, {self.cd4_cd8_ratio_bins[i+1]})"
            explanation.append(f"  Bin {i}: {bin_range:20s} - {interp}")
        explanation.append("")
        
        explanation.append("ACTIVATION MARKER BINS (HLA-DR Expression):")
        explanation.append("-" * 80)
        for i, interp in enumerate(self.ACTIVATION_INTERPRETATIONS):
            bin_range = f"[{self.activation_bins[i]}, {self.activation_bins[i+1]})"
            explanation.append(f"  Bin {i}: {bin_range:20s} - {interp}")
        explanation.append("")
        
        explanation.append("=" * 80)
        
        return "\n".join(explanation)
