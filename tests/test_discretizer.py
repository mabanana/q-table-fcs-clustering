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
        self.discretizer = ClinicalDiscretizer()
    
    def test_initialization(self):
        """Test discretizer initialization with default bins."""
        assert self.discretizer.cd4_bins == ClinicalDiscretizer.DEFAULT_CD4_BINS
        assert self.discretizer.cd4_cd8_ratio_bins == ClinicalDiscretizer.DEFAULT_CD4_CD8_RATIO_BINS
        assert self.discretizer.activation_bins == ClinicalDiscretizer.DEFAULT_ACTIVATION_BINS
    
    def test_custom_bins(self):
        """Test initialization with custom bin boundaries."""
        custom_cd4_bins = [0, 100, 300, 600, np.inf]
        discretizer = ClinicalDiscretizer(cd4_bins=custom_cd4_bins)
        assert discretizer.cd4_bins == custom_cd4_bins
    
    def test_discretize_cd4_basic(self):
        """Test CD4 discretization with basic values."""
        cd4_values = np.array([100, 250, 400, 600])
        bins = self.discretizer.discretize_cd4(cd4_values)
        
        # Expected: [0, 1, 2, 3] based on WHO criteria
        assert bins[0] == 0  # <200
        assert bins[1] == 1  # 200-350
        assert bins[2] == 2  # 350-500
        assert bins[3] == 3  # >500
    
    def test_discretize_cd4_boundary_cases(self):
        """Test CD4 discretization at bin boundaries."""
        cd4_values = np.array([200, 350, 500])
        bins = self.discretizer.discretize_cd4(cd4_values)
        
        # digitize with right=False means intervals [a, b)
        assert bins[0] == 1  # 200 falls into bin 1
        assert bins[1] == 2  # 350 falls into bin 2
        assert bins[2] == 3  # 500 falls into bin 3
    
    def test_discretize_cd4_cd8_ratio(self):
        """Test CD4/CD8 ratio discretization."""
        ratio_values = np.array([0.5, 1.2, 2.0, 3.0])
        bins = self.discretizer.discretize_cd4_cd8_ratio(ratio_values)
        
        assert bins[0] == 0  # <1.0 (inverted)
        assert bins[1] == 1  # 1.0-1.5
        assert bins[2] == 2  # 1.5-2.5
        assert bins[3] == 3  # >2.5
    
    def test_discretize_activation(self):
        """Test activation marker discretization."""
        activation_values = np.array([300, 800, 2000, 4000])
        bins = self.discretizer.discretize_activation(activation_values)
        
        assert bins[0] == 0  # <500
        assert bins[1] == 1  # 500-1500
        assert bins[2] == 2  # 1500-3000
        assert bins[3] == 3  # >3000
    
    def test_discretize_data(self):
        """Test discretization of a full DataFrame."""
        data = pd.DataFrame({
            'CD4': [150, 300, 450, 600],
            'CD8': [300, 200, 300, 200],
            'HLA-DR': [400, 1000, 2000, 5000]
        })
        
        result = self.discretizer.discretize_data(data, activation_column='HLA-DR')
        
        # Check that discretized columns were added
        assert 'cd4_bin' in result.columns
        assert 'cd4_cd8_ratio' in result.columns
        assert 'cd4_cd8_ratio_bin' in result.columns
        assert 'activation_bin' in result.columns
        
        # Check CD4 bins
        assert result['cd4_bin'].iloc[0] == 0  # 150 < 200
        assert result['cd4_bin'].iloc[1] == 1  # 200 <= 300 < 350
        assert result['cd4_bin'].iloc[2] == 2  # 350 <= 450 < 500
        assert result['cd4_bin'].iloc[3] == 3  # 600 >= 500
        
        # Check activation bins
        assert result['activation_bin'].iloc[0] == 0  # 400 < 500
        assert result['activation_bin'].iloc[3] == 3  # 5000 > 3000
    
    def test_discretize_data_no_activation(self):
        """Test discretization without activation marker."""
        data = pd.DataFrame({
            'CD4': [150, 300],
            'CD8': [300, 200]
        })
        
        result = self.discretizer.discretize_data(data, activation_column=None)
        
        assert 'cd4_bin' in result.columns
        assert 'cd4_cd8_ratio_bin' in result.columns
        assert 'activation_bin' not in result.columns
    
    def test_create_state_2_features(self):
        """Test state creation with 2 features (CD4, CD4/CD8 ratio)."""
        # State encoding: cd4_bin * 4 + cd4_cd8_ratio_bin
        state1 = self.discretizer.create_state(0, 0, None)
        assert state1 == 0  # 0*4 + 0
        
        state2 = self.discretizer.create_state(1, 2, None)
        assert state2 == 6  # 1*4 + 2
        
        state3 = self.discretizer.create_state(3, 3, None)
        assert state3 == 15  # 3*4 + 3
    
    def test_create_state_3_features(self):
        """Test state creation with 3 features (including activation)."""
        # State encoding: cd4_bin * 16 + cd4_cd8_ratio_bin * 4 + activation_bin
        state1 = self.discretizer.create_state(0, 0, 0)
        assert state1 == 0  # 0*16 + 0*4 + 0
        
        state2 = self.discretizer.create_state(1, 2, 1)
        assert state2 == 25  # 1*16 + 2*4 + 1
        
        state3 = self.discretizer.create_state(3, 3, 3)
        assert state3 == 63  # 3*16 + 3*4 + 3
    
    def test_decode_state_2_features(self):
        """Test state decoding for 2 features."""
        cd4, ratio, activation = self.discretizer.decode_state(6, num_features=2)
        
        assert cd4 == 1
        assert ratio == 2
        assert activation is None
    
    def test_decode_state_3_features(self):
        """Test state decoding for 3 features."""
        cd4, ratio, activation = self.discretizer.decode_state(25, num_features=3)
        
        assert cd4 == 1
        assert ratio == 2
        assert activation == 1
    
    def test_state_roundtrip(self):
        """Test that encoding and decoding are inverses."""
        # 2 features
        original_cd4, original_ratio = 2, 1
        state = self.discretizer.create_state(original_cd4, original_ratio, None)
        decoded_cd4, decoded_ratio, decoded_activation = self.discretizer.decode_state(state, num_features=2)
        
        assert decoded_cd4 == original_cd4
        assert decoded_ratio == original_ratio
        assert decoded_activation is None
        
        # 3 features
        original_cd4, original_ratio, original_activation = 2, 1, 3
        state = self.discretizer.create_state(original_cd4, original_ratio, original_activation)
        decoded_cd4, decoded_ratio, decoded_activation = self.discretizer.decode_state(state, num_features=3)
        
        assert decoded_cd4 == original_cd4
        assert decoded_ratio == original_ratio
        assert decoded_activation == original_activation
    
    def test_get_state_from_data(self):
        """Test state extraction from discretized DataFrame."""
        data = pd.DataFrame({
            'cd4_bin': [0, 1, 2, 3],
            'cd4_cd8_ratio_bin': [0, 1, 2, 3]
        })
        
        states = self.discretizer.get_state_from_data(data, use_activation=False)
        
        expected_states = [0, 5, 10, 15]  # cd4*4 + ratio
        np.testing.assert_array_equal(states, expected_states)
    
    def test_get_clinical_interpretation(self):
        """Test clinical interpretation generation."""
        interpretation = self.discretizer.get_clinical_interpretation(0, 0, None)
        
        assert "Severe immunodeficiency" in interpretation
        assert "Inverted" in interpretation
    
    def test_get_bin_statistics(self):
        """Test bin statistics calculation."""
        data = pd.DataFrame({
            'cd4_bin': [0, 0, 1, 1, 2, 3],
            'cd4_cd8_ratio_bin': [0, 1, 1, 2, 2, 3]
        })
        
        stats = self.discretizer.get_bin_statistics(data)
        
        assert 'cd4_bin_counts' in stats
        assert 'cd4_cd8_ratio_bin_counts' in stats
        assert stats['cd4_bin_counts'][0] == 2
        assert stats['cd4_bin_counts'][1] == 2
    
    def test_explain_discretization(self):
        """Test discretization explanation generation."""
        explanation = self.discretizer.explain_discretization()
        
        assert "CLINICALLY-INFORMED DISCRETIZATION" in explanation
        assert "WHO HIV Staging" in explanation
        assert "CD4 COUNT BINS" in explanation
        assert "CD4/CD8 RATIO BINS" in explanation
    
    def test_edge_case_zero_values(self):
        """Test handling of zero values."""
        data = pd.DataFrame({
            'CD4': [0, 0],
            'CD8': [0, 100]
        })
        
        # Should handle division by zero gracefully
        result = self.discretizer.discretize_data(data)
        
        assert 'cd4_cd8_ratio' in result.columns
        # Check that we didn't get NaN or inf
        assert not result['cd4_cd8_ratio'].isna().any()
        assert not np.isinf(result['cd4_cd8_ratio']).any()
    
    def test_edge_case_very_large_values(self):
        """Test handling of very large values."""
        cd4_values = np.array([1e6, 1e7])
        bins = self.discretizer.discretize_cd4(cd4_values)
        
        # Should all fall into highest bin
        assert all(bins == 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
