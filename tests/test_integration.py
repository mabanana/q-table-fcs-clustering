"""
Integration tests for preprocessing → training pipeline.

Tests the end-to-end compensation workflow to ensure that:
1. Compensation is properly applied and preserved during preprocessing
2. FCS loader correctly identifies compensated files
3. Compensation matrices are consistently used across pipeline stages
"""

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import CompensationMatrixLoader, FCSPreprocessor, MetadataRouter
from src.fcs_loader import FCSLoader


class TestCompensationIntegration:
    """Test end-to-end preprocessing and loading workflow."""
    
    def test_compensation_matrix_loader(self):
        """Test that CompensationMatrixLoader works correctly."""
        loader = CompensationMatrixLoader()
        
        # Test with a real compensation file if it exists
        comp_file = "data/compensation/0001.csv"
        if os.path.exists(comp_file):
            matrix = loader.load_from_csv(comp_file)
            assert matrix is not None
            assert matrix.shape[0] == matrix.shape[1]  # Should be square
            assert loader.markers is not None
            assert len(loader.markers) == matrix.shape[0]
    
    def test_fcs_loader_initialization_with_compensation(self):
        """Test that FCSLoader initializes correctly with compensation."""
        # Test without compensation
        loader1 = FCSLoader(markers=["CD4", "CD8"])
        assert loader1.compensation_matrix is None
        assert loader1.preprocessor is None
        
        # Test with compensation (if file exists)
        comp_file = "data/compensation/0001.csv"
        if os.path.exists(comp_file):
            loader2 = FCSLoader.from_compensation_file(
                compensation_csv=comp_file,
                markers=["CD4", "CD8"]  # Explicitly pass markers
            )
            assert loader2.compensation_matrix is not None
            assert loader2.preprocessor is not None
            assert loader2.compensation_markers is not None
    
    def test_compensation_matrix_consistency(self):
        """Test that same compensation matrix used in preprocessing and loading."""
        comp_csv = "data/compensation/0001.csv"
        
        if not os.path.exists(comp_csv):
            pytest.skip("Compensation file not found")
        
        # Load via preprocessing
        loader1 = CompensationMatrixLoader()
        matrix1 = loader1.load_from_csv(comp_csv)
        
        # Load via fcs_loader factory
        fcs_loader = FCSLoader.from_compensation_file(comp_csv)
        matrix2 = fcs_loader.compensation_matrix
        
        # Should be identical
        assert np.allclose(matrix1, matrix2)
        assert loader1.markers == fcs_loader.compensation_markers
    
    def test_fcs_preprocessor_fallback_method(self):
        """Test that FCSPreprocessor fallback method works."""
        preprocessor = FCSPreprocessor()
        
        # Create a simple DataFrame for testing
        test_df = pd.DataFrame({
            'CD4': [100, 200, 300],
            'CD8': [50, 100, 150]
        })
        
        # Test fallback method
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'test_input.fcs')
            output_path = os.path.join(tmpdir, 'test_output.fcs')
            
            # Create a dummy input file
            with open(input_path, 'w') as f:
                f.write('dummy')
            
            # Test fallback
            result = preprocessor._fallback_copy_method(
                input_path,
                output_path,
                test_df
            )
            
            assert result is True
            assert os.path.exists(output_path)
            
            # Check CSV was created
            csv_path = output_path.replace('.fcs', '_compensated.csv')
            assert os.path.exists(csv_path)
            
            # Verify CSV content
            saved_df = pd.read_csv(csv_path)
            assert list(saved_df.columns) == ['CD4', 'CD8']
            assert len(saved_df) == 3
    
    def test_metadata_router_initialization(self):
        """Test MetadataRouter initialization and basic functionality."""
        router = MetadataRouter(base_output_dir="data/")
        assert router.base_output_dir == "data/"
        assert router.metadata is None
        assert router.routing_map == {}
    
    def test_fcs_loader_files_loaded_tracking(self):
        """Test that FCSLoader tracks loaded files."""
        loader = FCSLoader(markers=["CD4", "CD8"])
        assert hasattr(loader, 'files_loaded')
        assert isinstance(loader.files_loaded, list)
        assert len(loader.files_loaded) == 0


class TestPreprocessingWorkflow:
    """Test preprocessing workflow components."""
    
    def test_compensation_apply_workflow(self):
        """Test the compensation application workflow."""
        # Create a simple compensation matrix
        comp_matrix = np.array([
            [1.0, 0.1],
            [0.05, 1.0]
        ])
        markers = ['CD4', 'CD8']
        
        # Create test data
        test_data = pd.DataFrame({
            'CD4': [100.0, 200.0, 300.0],
            'CD8': [50.0, 100.0, 150.0]
        })
        
        # Apply compensation
        preprocessor = FCSPreprocessor()
        compensated = preprocessor.apply_compensation(
            test_data,
            comp_matrix,
            markers
        )
        
        # Verify output
        assert isinstance(compensated, pd.DataFrame)
        assert list(compensated.columns) == ['CD4', 'CD8']
        assert len(compensated) == 3
        
        # Verify compensation was applied (data should be different)
        assert not np.allclose(compensated.values, test_data.values)
    
    def test_compensation_with_missing_markers(self):
        """Test compensation handling when markers are missing."""
        comp_matrix = np.array([
            [1.0, 0.1],
            [0.05, 1.0]
        ])
        markers = ['CD4', 'CD3']  # CD3 not in data
        
        test_data = pd.DataFrame({
            'CD4': [100.0, 200.0, 300.0],
            'CD8': [50.0, 100.0, 150.0]
        })
        
        preprocessor = FCSPreprocessor()
        
        # Should raise ValueError for missing markers
        with pytest.raises(ValueError, match="Could not match all markers"):
            preprocessor.apply_compensation(test_data, comp_matrix, markers)


class TestFCSLoaderWorkflow:
    """Test FCS loader workflow."""
    
    def test_loader_compensation_parameters(self):
        """Test FCSLoader with compensation parameters."""
        markers = ["CD4", "CD8"]
        comp_matrix = np.eye(2)  # Identity matrix
        comp_markers = ["CD4", "CD8"]
        
        loader = FCSLoader(
            markers=markers,
            compensation_matrix=comp_matrix,
            compensation_markers=comp_markers
        )
        
        assert loader.markers == markers
        assert np.allclose(loader.compensation_matrix, comp_matrix)
        assert loader.compensation_markers == comp_markers
        assert loader.preprocessor is not None
    
    def test_loader_without_compensation(self):
        """Test FCSLoader without compensation."""
        markers = ["CD4", "CD8"]
        
        loader = FCSLoader(markers=markers)
        
        assert loader.markers == markers
        assert loader.compensation_matrix is None
        assert loader.compensation_markers is None
        assert loader.preprocessor is None
