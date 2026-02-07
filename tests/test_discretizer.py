"""
Unit tests for the Clinical Discretizer module.
"""

import pytest
import numpy as np
import pandas as pd
from src.discretizer import ClinicalDiscretizer


class TestClinicalDiscretizer:
    """Test suite for ClinicalDiscretizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Test with quantile method
        self.discretizer_quantile = ClinicalDiscretizer(
            method="quantile",
            n_bins=4,
            selected_markers=["CD123", "IFNa", "IL12"]
        )
        
        # Test with fixed method
        self.discretizer_fixed = ClinicalDiscretizer(
            method="fixed",
            n_bins=4,
            selected_markers=["CD123", "IFNa", "IL12"],
            cytokine_bins=[0, 100, 500, 2000, np.inf],
            dendritic_bins=[0, 500, 1500, 5000, np.inf]
        )
    
    def test_initialization_quantile(self):
        """Test discretizer initialization with quantile method."""
        assert self.discretizer_quantile.method == "quantile"
        assert self.discretizer_quantile.n_bins == 4
        assert self.discretizer_quantile.selected_markers == ["CD123", "IFNa", "IL12"]
    
    def test_initialization_fixed(self):
        """Test discretizer initialization with fixed method."""
        assert self.discretizer_fixed.method == "fixed"
        assert self.discretizer_fixed.cytokine_bins == [0, 100, 500, 2000, np.inf]
        assert self.discretizer_fixed.dendritic_bins == [0, 500, 1500, 5000, np.inf]
    
    def test_discretize_marker_quantile(self):
        """Test quantile-based marker discretization."""
        # Create test data with known distribution
        values = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000])
        bins = self.discretizer_quantile.discretize_marker(values, "IFNa")
        
        # Should have 4 bins (0-3)
        assert bins.min() >= 0
        assert bins.max() <= 3
        assert len(bins) == len(values)
    
    def test_discretize_marker_fixed_cytokine(self):
        """Test fixed binning for cytokine markers."""
        # Test IFNa (cytokine)
        values = np.array([50, 300, 1000, 3000])
        bins = self.discretizer_fixed.discretize_marker(values, "IFNa")
        
        # Expected: [0, 1, 2, 3] based on cytokine bins
        assert bins[0] == 0  # 50 < 100
        assert bins[1] == 1  # 100 <= 300 < 500
        assert bins[2] == 2  # 500 <= 1000 < 2000
        assert bins[3] == 3  # 3000 >= 2000
    
    def test_discretize_marker_fixed_dendritic(self):
        """Test fixed binning for dendritic cell markers."""
        # Test CD123 (dendritic)
        values = np.array([200, 800, 3000, 6000])
        bins = self.discretizer_fixed.discretize_marker(values, "CD123")
        
        # Expected based on dendritic bins
        assert bins[0] == 0  # 200 < 500
        assert bins[1] == 1  # 500 <= 800 < 1500
        assert bins[2] == 2  # 1500 <= 3000 < 5000
        assert bins[3] == 3  # 6000 >= 5000
    
    def test_discretize_data_quantile(self):
        """Test discretization of a full DataFrame with quantile method."""
        data = pd.DataFrame({
            'IFNa': [10, 50, 100, 500, 1000, 2000, 5000, 10000],
            'CD123': [100, 300, 600, 1000, 2000, 3000, 5000, 8000],
            'IL12': [5, 20, 80, 300, 800, 1500, 3000, 6000]
        })
        
        result = self.discretizer_quantile.discretize_data(
            data,
            markers=["IFNa", "CD123", "IL12"]
        )
        
        # Check that discretized columns were added
        assert 'IFNa_bin' in result.columns
        assert 'CD123_bin' in result.columns
        assert 'IL12_bin' in result.columns
        
        # Check that bins are in valid range
        for marker in ["IFNa", "CD123", "IL12"]:
            bin_col = f"{marker}_bin"
            assert result[bin_col].min() >= 0
            assert result[bin_col].max() <= 3
    
    def test_discretize_data_fixed(self):
        """Test discretization with fixed bins."""
        data = pd.DataFrame({
            'IFNa': [50, 300, 1000, 3000],
            'CD123': [200, 800, 3000, 6000],
            'IL12': [50, 300, 1000, 3000]
        })
        
        result = self.discretizer_fixed.discretize_data(
            data,
            markers=["IFNa", "CD123", "IL12"]
        )
        
        # Check IFNa bins (cytokine)
        assert result['IFNa_bin'].iloc[0] == 0  # 50 < 100
        assert result['IFNa_bin'].iloc[3] == 3  # 3000 >= 2000
        
        # Check CD123 bins (dendritic)
        assert result['CD123_bin'].iloc[0] == 0  # 200 < 500
        assert result['CD123_bin'].iloc[3] == 3  # 6000 >= 5000
    
    def test_create_state_3_markers(self):
        """Test state creation with 3 markers."""
        # State encoding: CD123_bin * 16 + IFNa_bin * 4 + IL12_bin
        marker_bins = {'CD123': 0, 'IFNa': 0, 'IL12': 0}
        state1 = self.discretizer_quantile.create_state(marker_bins)
        assert state1 == 0  # 0*16 + 0*4 + 0
        
        marker_bins = {'CD123': 1, 'IFNa': 2, 'IL12': 3}
        state2 = self.discretizer_quantile.create_state(marker_bins)
        assert state2 == 27  # 1*16 + 2*4 + 3
        
        marker_bins = {'CD123': 3, 'IFNa': 3, 'IL12': 3}
        state3 = self.discretizer_quantile.create_state(marker_bins)
        assert state3 == 63  # 3*16 + 3*4 + 3
    
    def test_decode_state_3_markers(self):
        """Test state decoding for 3 markers."""
        marker_bins = self.discretizer_quantile.decode_state(27)
        
        assert marker_bins['CD123'] == 1
        assert marker_bins['IFNa'] == 2
        assert marker_bins['IL12'] == 3
    
    def test_state_roundtrip(self):
        """Test that encoding and decoding are inverses."""
        original_bins = {'CD123': 2, 'IFNa': 1, 'IL12': 3}
        state = self.discretizer_quantile.create_state(original_bins)
        decoded_bins = self.discretizer_quantile.decode_state(state)
        
        assert decoded_bins == original_bins
    
    def test_get_state_from_data(self):
        """Test state extraction from discretized DataFrame."""
        data = pd.DataFrame({
            'CD123_bin': [0, 1, 2, 3],
            'IFNa_bin': [0, 1, 2, 3],
            'IL12_bin': [0, 1, 2, 3]
        })
        
        states = self.discretizer_quantile.get_state_from_data(data)
        
        # Expected states: CD123*16 + IFNa*4 + IL12
        expected_states = [0, 21, 42, 63]  # 0*16+0*4+0, 1*16+1*4+1, 2*16+2*4+2, 3*16+3*4+3
        np.testing.assert_array_equal(states, expected_states)
    
    def test_get_clinical_interpretation(self):
        """Test clinical interpretation generation."""
        marker_bins = {'CD123': 0, 'IFNa': 1, 'IL12': 2}
        interpretation = self.discretizer_quantile.get_clinical_interpretation(marker_bins)
        
        assert "CD123" in interpretation
        assert "IFNa" in interpretation
        assert "IL12" in interpretation
    
    def test_get_bin_statistics(self):
        """Test bin statistics calculation."""
        data = pd.DataFrame({
            'CD123_bin': [0, 0, 1, 1, 2, 3],
            'IFNa_bin': [0, 1, 1, 2, 2, 3],
            'IL12_bin': [0, 0, 1, 2, 3, 3]
        })
        
        stats = self.discretizer_quantile.get_bin_statistics(data)
        
        assert 'CD123_bin_counts' in stats
        assert 'IFNa_bin_counts' in stats
        assert 'IL12_bin_counts' in stats
        assert stats['CD123_bin_counts'][0] == 2
        assert stats['CD123_bin_counts'][1] == 2
    
    def test_explain_discretization_quantile(self):
        """Test discretization explanation generation for quantile method."""
        explanation = self.discretizer_quantile.explain_discretization()
        
        assert "DENDRITIC CELL & CYTOKINE DISCRETIZATION" in explanation
        assert "QUANTILE-BASED BINNING" in explanation
        assert "CD123, IFNa, IL12" in explanation
    
    def test_explain_discretization_fixed(self):
        """Test discretization explanation generation for fixed method."""
        explanation = self.discretizer_fixed.explain_discretization()
        
        assert "DENDRITIC CELL & CYTOKINE DISCRETIZATION" in explanation
        assert "FIXED FLUORESCENCE INTENSITY BINS" in explanation
        assert "CYTOKINE MARKERS" in explanation
        assert "DENDRITIC CELL MARKERS" in explanation
    
    def test_edge_case_uniform_values(self):
        """Test handling of uniform values in quantile method."""
        # All same values
        values = np.array([100, 100, 100, 100])
        bins = self.discretizer_quantile.discretize_marker(values, "IFNa")
        
        # Should handle gracefully (all in same bin)
        assert len(np.unique(bins)) == 1
    
    def test_edge_case_very_large_values(self):
        """Test handling of very large fluorescence values."""
        values = np.array([1e6, 1e7, 1e8])
        bins = self.discretizer_fixed.discretize_marker(values, "IFNa")
        
        # Should all fall into highest bin (3)
        assert all(bins == 3)
    
    def test_marker_type_classification(self):
        """Test that markers are correctly classified as cytokine or dendritic."""
        # Cytokine markers should use cytokine bins
        assert "IFNa" in ClinicalDiscretizer.CYTOKINE_MARKERS
        assert "IL12" in ClinicalDiscretizer.CYTOKINE_MARKERS
        
        # Dendritic markers should use dendritic bins
        assert "CD123" in ClinicalDiscretizer.DENDRITIC_MARKERS
        assert "CD11c" in ClinicalDiscretizer.DENDRITIC_MARKERS
    
    def test_fixed_binning_uses_correct_bins(self):
        """Test that cytokine markers use cytokine bins and dendritic markers use dendritic bins."""
        # Test that IFNa (cytokine) uses cytokine bins
        # Cytokine bins: [0, 100, 500, 2000, inf]
        ifna_value = np.array([150])  # Should fall in bin 1 (100-500)
        ifna_bin = self.discretizer_fixed.discretize_marker(ifna_value, "IFNa")
        assert ifna_bin[0] == 1
        
        # Test that same value for CD123 (dendritic) uses different bins
        # Dendritic bins: [0, 500, 1500, 5000, inf]
        cd123_value = np.array([150])  # Should fall in bin 0 (0-500)
        cd123_bin = self.discretizer_fixed.discretize_marker(cd123_value, "CD123")
        assert cd123_bin[0] == 0
        
        # Verify they use different bins for the same numerical value
        assert ifna_bin[0] != cd123_bin[0]
        
        # Test boundary values
        boundary_value = np.array([500])
        ifna_boundary = self.discretizer_fixed.discretize_marker(boundary_value, "IFNa")
        cd123_boundary = self.discretizer_fixed.discretize_marker(boundary_value, "CD123")
        assert ifna_boundary[0] == 2  # 500 in cytokine bins (500-2000)
        assert cd123_boundary[0] == 1  # 500 in dendritic bins (500-1500)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
