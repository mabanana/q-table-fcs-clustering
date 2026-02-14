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
        self.feature_bins = {
            "IFNa": [0, 0.5, 1.0, 2.0, 3.0, np.inf],
            "TNFa": [0, 0.5, 1.0, 2.0, 3.0, np.inf]
        }
        self.state_features = ["IFNa", "TNFa"]
        self.discretizer = ClinicalDiscretizer(
            feature_bins=self.feature_bins,
            state_features=self.state_features
        )
    
    def test_initialization(self):
        """Test discretizer initialization with default bins."""
        assert self.discretizer.feature_bins == self.feature_bins
        assert self.discretizer.state_features == self.state_features
    
    def test_custom_bins(self):
        """Test initialization with custom bin boundaries."""
        custom_bins = {"IFNa": [0, 1, 2, 3, np.inf]}
        discretizer = ClinicalDiscretizer(feature_bins=custom_bins)
        assert discretizer.feature_bins == custom_bins
    
    def test_discretize_feature_basic(self):
        """Test feature discretization with basic values."""
        values = np.array([0.1, 0.6, 1.2, 3.0, 4.0])
        bins = self.discretizer.discretize_feature(values, self.feature_bins["IFNa"])

        assert bins[0] == 0
        assert bins[1] == 1
        assert bins[2] == 2
        assert bins[3] == 3
        assert bins[4] == 4
    
    def test_discretize_feature_boundary_cases(self):
        """Test discretization at bin boundaries."""
        values = np.array([0.5, 1.0, 2.0, 3.0])
        bins = self.discretizer.discretize_feature(values, self.feature_bins["IFNa"])

        assert bins[0] == 1
        assert bins[1] == 2
        assert bins[2] == 3
        assert bins[3] == 4
    
    def test_discretize_data(self):
        """Test discretization of a full DataFrame."""
        data = pd.DataFrame({
            'IFNa': [0.1, 0.8, 1.4, 3.5],
            'TNFa': [0.2, 0.7, 1.9, 2.5]
        })

        result = self.discretizer.discretize_data(data, feature_names=self.state_features)

        assert 'IFNa_bin' in result.columns
        assert 'TNFa_bin' in result.columns
        assert result['IFNa_bin'].iloc[0] == 0
        assert result['IFNa_bin'].iloc[1] == 1
        assert result['TNFa_bin'].iloc[2] == 2
    
    def test_create_state_2_features(self):
        """Test state creation with 2 features."""
        state1 = self.discretizer.create_state([0, 0])
        assert state1 == 0  # 0*5 + 0
        
        state2 = self.discretizer.create_state([1, 2])
        assert state2 == 7  # 1*5 + 2
        
        state3 = self.discretizer.create_state([3, 3])
        assert state3 == 18  # 3*5 + 3
    
    def test_create_state_3_features(self):
        """Test state creation with 3 features."""
        state1 = self.discretizer.create_state([0, 0, 0])
        assert state1 == 0  # 0*25 + 0*5 + 0
        
        state2 = self.discretizer.create_state([1, 2, 1])
        assert state2 == 36  # 1*25 + 2*5 + 1
        
        state3 = self.discretizer.create_state([3, 3, 3])
        assert state3 == 93  # 3*25 + 3*5 + 3
    
    def test_decode_state_2_features(self):
        """Test state decoding for 2 features."""
        bins = self.discretizer.decode_state(7, num_features=2, bin_sizes=[5, 5])
        
        assert bins == (1, 2)
    
    def test_decode_state_3_features(self):
        """Test state decoding for 3 features."""
        bins = self.discretizer.decode_state(36, num_features=3, bin_sizes=[5, 5, 5])
        
        assert bins == (1, 2, 1)
    
    def test_state_roundtrip(self):
        """Test that encoding and decoding are inverses."""
        # 2 features
        original_bins = [2, 1]
        state = self.discretizer.create_state(original_bins, bin_sizes=[5, 5])
        decoded_bins = self.discretizer.decode_state(state, num_features=2, bin_sizes=[5, 5])
        
        assert decoded_bins == tuple(original_bins)
        
        # 3 features
        original_bins = [2, 1, 3]
        state = self.discretizer.create_state(original_bins, bin_sizes=[5, 5, 5])
        decoded_bins = self.discretizer.decode_state(state, num_features=3, bin_sizes=[5, 5, 5])
        
        assert decoded_bins == tuple(original_bins)
    
    def test_get_state_from_data(self):
        """Test state extraction from discretized DataFrame."""
        data = pd.DataFrame({
            'IFNa_bin': [0, 1, 2, 3],
            'TNFa_bin': [0, 1, 2, 3]
        })
        
        states = self.discretizer.get_state_from_data(data, feature_names=self.state_features)
        
        expected_states = [0, 6, 12, 18]  # bin1*5 + bin2
        np.testing.assert_array_equal(states, expected_states)
    
    def test_get_clinical_interpretation(self):
        """Test clinical interpretation generation."""
        interpretation = self.discretizer.get_clinical_interpretation(
            [0, 1],
            feature_names=self.state_features
        )
        
        assert "IFNa" in interpretation
        assert "TNFa" in interpretation
    
    def test_get_bin_statistics(self):
        """Test bin statistics calculation."""
        data = pd.DataFrame({
            'IFNa_bin': [0, 0, 1, 1, 2, 3],
            'TNFa_bin': [0, 1, 1, 2, 2, 3]
        })
        
        stats = self.discretizer.get_bin_statistics(data, feature_names=self.state_features)
        
        assert 'IFNa_bin_counts' in stats
        assert 'TNFa_bin_counts' in stats
        assert stats['IFNa_bin_counts'][0] == 2
        assert stats['IFNa_bin_counts'][1] == 2
    
    def test_explain_discretization(self):
        """Test discretization explanation generation."""
        explanation = self.discretizer.explain_discretization()
        
        assert "FEATURE DISCRETIZATION" in explanation
        assert "IFNa" in explanation
        assert "TNFa" in explanation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
