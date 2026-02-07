"""
Clinically-Informed Discretizer Module

Implements discretization of continuous flow cytometry data into discrete bins
for dendritic cell markers and cytokines (FlowCap II dataset).
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalDiscretizer:
    """
    Discretizes continuous flow cytometry data for dendritic cell and cytokine markers.
    
    Supports two discretization methods:
    - Quantile-based: Data-driven bins using quantiles (adaptive to data distribution)
    - Fixed: Predefined fluorescence intensity thresholds
    """
    
    # Default bin boundaries for fixed method (fluorescence intensity)
    DEFAULT_CYTOKINE_BINS = [0, 100, 500, 2000, np.inf]
    DEFAULT_DENDRITIC_BINS = [0, 500, 1500, 5000, np.inf]
    
    # Marker type classification
    CYTOKINE_MARKERS = ['IFNa', 'IL6', 'IL12', 'TNFa']
    DENDRITIC_MARKERS = ['CD123', 'MHCII', 'CD14', 'CD11c']
    
    def __init__(
        self,
        method: str = "quantile",
        n_bins: int = 4,
        selected_markers: Optional[List[str]] = None,
        cytokine_bins: Optional[List[float]] = None,
        dendritic_bins: Optional[List[float]] = None
    ):
        """
        Initialize the discretizer for dendritic cell/cytokine data.
        
        Args:
            method: Discretization method ("quantile" or "fixed")
            n_bins: Number of bins for quantile method (default: 4)
            selected_markers: Markers to use for state encoding (default: ["CD123", "IFNa", "IL12"])
            cytokine_bins: Bin boundaries for cytokines (used with fixed method)
            dendritic_bins: Bin boundaries for dendritic markers (used with fixed method)
        """
        self.method = method
        self.n_bins = n_bins
        self.selected_markers = selected_markers or ["CD123", "IFNa", "IL12"]
        self.cytokine_bins = cytokine_bins or self.DEFAULT_CYTOKINE_BINS
        self.dendritic_bins = dendritic_bins or self.DEFAULT_DENDRITIC_BINS
        
        # Store computed quantile bins (will be calculated from data)
        self.quantile_bins_cache = {}
        
        logger.info("Initialized ClinicalDiscretizer with:")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Number of bins: {self.n_bins}")
        logger.info(f"  Selected markers for state: {self.selected_markers}")
        if method == "fixed":
            logger.info(f"  Cytokine bins: {self.cytokine_bins}")
            logger.info(f"  Dendritic bins: {self.dendritic_bins}")
    
    def discretize_marker_quantile(self, values: np.ndarray, marker_name: str) -> np.ndarray:
        """
        Discretize marker values using quantile-based binning.
        
        Args:
            values: Array of marker values
            marker_name: Name of the marker (for caching)
            
        Returns:
            Array of bin indices (0 to n_bins-1)
        """
        # Calculate quantile boundaries
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        bins = np.quantile(values, quantiles)
        
        # Handle edge case where all values are the same
        if len(np.unique(bins)) == 1:
            return np.zeros(len(values), dtype=int)
        
        # Store bins for later reference
        self.quantile_bins_cache[marker_name] = bins
        
        # Digitize values into bins (bins are inclusive on the left, exclusive on the right)
        discretized = np.digitize(values, bins[1:-1], right=False)
        
        return discretized
    
    def discretize_marker_fixed(self, values: np.ndarray, marker_name: str) -> np.ndarray:
        """
        Discretize marker values using fixed fluorescence intensity thresholds.
        
        Args:
            values: Array of marker values
            marker_name: Name of the marker
            
        Returns:
            Array of bin indices (0 to 3)
        """
        # Determine which bin set to use based on marker type
        if marker_name in self.CYTOKINE_MARKERS:
            bins = self.cytokine_bins
        elif marker_name in self.DENDRITIC_MARKERS:
            bins = self.dendritic_bins
        else:
            # Warn about unknown marker and default to dendritic bins
            logger.warning(
                f"Unknown marker '{marker_name}' not classified as cytokine or dendritic. "
                f"Defaulting to dendritic cell bins. Known cytokines: {self.CYTOKINE_MARKERS}, "
                f"Known dendritic markers: {self.DENDRITIC_MARKERS}"
            )
            bins = self.dendritic_bins
        
        return np.digitize(values, bins[1:], right=False)
    
    def discretize_marker(self, values: np.ndarray, marker_name: str) -> np.ndarray:
        """
        Discretize marker values based on configured method.
        
        Args:
            values: Array of marker values
            marker_name: Name of the marker
            
        Returns:
            Array of bin indices
        """
        if self.method == "quantile":
            return self.discretize_marker_quantile(values, marker_name)
        elif self.method == "fixed":
            return self.discretize_marker_fixed(values, marker_name)
        else:
            raise ValueError(f"Unknown discretization method: {self.method}")
    
    def discretize_data(
        self,
        data: pd.DataFrame,
        markers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Discretize all relevant markers in a DataFrame.
        
        Args:
            data: DataFrame containing flow cytometry data
            markers: List of marker names to discretize (defaults to all available)
            
        Returns:
            DataFrame with discretized features added as *_bin columns
        """
        result = data.copy()
        
        # Use provided markers or detect from dataframe
        if markers is None:
            # Detect available markers from columns
            markers = []
            all_known_markers = self.CYTOKINE_MARKERS + self.DENDRITIC_MARKERS
            for marker in all_known_markers:
                if marker in result.columns:
                    markers.append(marker)
        
        # Discretize each marker
        for marker in markers:
            if marker in result.columns:
                bin_col_name = f"{marker}_bin"
                result[bin_col_name] = self.discretize_marker(
                    result[marker].values, marker
                )
                logger.debug(f"Discretized {marker} into {bin_col_name}")
            else:
                logger.warning(f"Marker {marker} not found in data columns")
        
        return result
    
    def create_state(self, marker_bins: Dict[str, int]) -> int:
        """
        Create a discrete state representation from marker bin indices.
        
        State encoding uses base-n_bins encoding where n_bins is typically 4.
        For selected markers [M1, M2, M3] with bins [b1, b2, b3]:
        state = b1 * (n_bins^2) + b2 * (n_bins^1) + b3 * (n_bins^0)
        
        Args:
            marker_bins: Dictionary mapping marker names to bin indices
            
        Returns:
            Integer state representation
        """
        state = 0
        multiplier = 1
        
        # Encode in reverse order (rightmost marker is least significant)
        for marker in reversed(self.selected_markers):
            bin_idx = marker_bins.get(marker, 0)
            state += bin_idx * multiplier
            multiplier *= self.n_bins
        
        return state
    
    def decode_state(self, state: int) -> Dict[str, int]:
        """
        Decode a state representation back into marker bin indices.
        
        Args:
            state: Integer state representation
            
        Returns:
            Dictionary mapping marker names to bin indices
        """
        marker_bins = {}
        
        for marker in reversed(self.selected_markers):
            bin_idx = state % self.n_bins
            marker_bins[marker] = bin_idx
            state //= self.n_bins
        
        return marker_bins
    
    def get_state_from_data(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Convert discretized data to state representations.
        
        Args:
            data: DataFrame with discretized features (*_bin columns)
            
        Returns:
            Array of state representations
        """
        states = []
        
        for idx, row in data.iterrows():
            marker_bins = {}
            for marker in self.selected_markers:
                bin_col = f"{marker}_bin"
                if bin_col in row:
                    marker_bins[marker] = int(row[bin_col])
                else:
                    logger.warning(f"Bin column {bin_col} not found, using 0")
                    marker_bins[marker] = 0
            
            state = self.create_state(marker_bins)
            states.append(state)
        
        return np.array(states)
    
    def get_marker_interpretation(self, marker: str, bin_idx: int) -> str:
        """
        Get interpretation for a marker bin.
        
        Args:
            marker: Marker name
            bin_idx: Bin index
            
        Returns:
            Human-readable interpretation
        """
        if self.method == "quantile":
            if marker in self.quantile_bins_cache:
                bins = self.quantile_bins_cache[marker]
                if bin_idx < len(bins) - 1:
                    return f"[{bins[bin_idx]:.1f}, {bins[bin_idx+1]:.1f})"
            else:
                logger.warning(
                    f"Marker '{marker}' not found in quantile bins cache. "
                    f"Discretization may not have been performed yet."
                )
            return f"Bin {bin_idx}"
        else:  # fixed method
            if marker in self.CYTOKINE_MARKERS:
                bins = self.cytokine_bins
                labels = ["Low", "Medium", "High", "Very High"]
            else:
                bins = self.dendritic_bins
                labels = ["Low", "Medium", "High", "Very High"]
            
            if bin_idx < len(labels):
                range_str = f"[{bins[bin_idx]}, {bins[bin_idx+1]})"
                return f"{labels[bin_idx]} {range_str}"
            return f"Bin {bin_idx}"
    
    def get_clinical_interpretation(self, marker_bins: Dict[str, int]) -> str:
        """
        Get clinical interpretation for a combination of marker bins.
        
        Args:
            marker_bins: Dictionary mapping marker names to bin indices
            
        Returns:
            Human-readable clinical interpretation
        """
        interpretation_parts = []
        
        for marker in self.selected_markers:
            if marker in marker_bins:
                bin_idx = marker_bins[marker]
                interp = self.get_marker_interpretation(marker, bin_idx)
                interpretation_parts.append(f"{marker}: {interp}")
        
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
        
        for marker in self.selected_markers:
            bin_col = f"{marker}_bin"
            if bin_col in data.columns:
                stats[f"{marker}_bin_counts"] = data[bin_col].value_counts().to_dict()
                stats[f"{marker}_bin_distribution"] = (
                    data[bin_col].value_counts(normalize=True).to_dict()
                )
        
        return stats
    
    def explain_discretization(self) -> str:
        """
        Generate a detailed explanation of the discretization scheme.
        
        Returns:
            Formatted string explaining the discretization approach
        """
        explanation = []
        explanation.append("=" * 80)
        explanation.append("DENDRITIC CELL & CYTOKINE DISCRETIZATION SCHEME")
        explanation.append("=" * 80)
        explanation.append("")
        
        explanation.append(f"Discretization Method: {self.method.upper()}")
        explanation.append(f"Number of bins: {self.n_bins}")
        explanation.append(f"Selected markers for state space: {', '.join(self.selected_markers)}")
        explanation.append("")
        
        if self.method == "quantile":
            explanation.append("QUANTILE-BASED BINNING (Data-Driven):")
            explanation.append("-" * 80)
            explanation.append(f"Each marker is divided into {self.n_bins} bins based on data quantiles.")
            explanation.append("Bins adapt to the actual distribution of fluorescence intensities.")
            explanation.append("Ensures balanced representation across all intensity levels.")
        else:
            explanation.append("FIXED FLUORESCENCE INTENSITY BINS:")
            explanation.append("-" * 80)
            explanation.append("")
            explanation.append("CYTOKINE MARKERS (IFNa, IL6, IL12, TNFa):")
            for i in range(len(self.cytokine_bins) - 1):
                lower = self.cytokine_bins[i]
                upper = self.cytokine_bins[i + 1]
                label = ["Low", "Medium", "High", "Very High"][i] if i < 4 else f"Bin {i}"
                explanation.append(f"  Bin {i}: [{lower}, {upper}) - {label}")
            explanation.append("")
            
            explanation.append("DENDRITIC CELL MARKERS (CD123, MHCII, CD14, CD11c):")
            for i in range(len(self.dendritic_bins) - 1):
                lower = self.dendritic_bins[i]
                upper = self.dendritic_bins[i + 1]
                label = ["Low", "Medium", "High", "Very High"][i] if i < 4 else f"Bin {i}"
                explanation.append(f"  Bin {i}: [{lower}, {upper}) - {label}")
        
        explanation.append("")
        explanation.append("STATE SPACE ENCODING:")
        explanation.append("-" * 80)
        n_states = self.n_bins ** len(self.selected_markers)
        explanation.append(f"Total states: {n_states} ({self.n_bins}^{len(self.selected_markers)} combinations)")
        explanation.append(f"State = {' + '.join([f'{m}_bin × {self.n_bins}^{i}' for i, m in enumerate(reversed(self.selected_markers))])}")
        explanation.append("")
        
        explanation.append("=" * 80)
        
        return "\n".join(explanation)
