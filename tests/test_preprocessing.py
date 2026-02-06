"""
Unit tests for the FCS Preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.preprocessing import (
    CompensationMatrixLoader,
    FCSPreprocessor,
    MetadataRouter
)


class TestCompensationMatrixLoader:
    """Test suite for CompensationMatrixLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = CompensationMatrixLoader()
        
        # Create a temporary compensation matrix CSV
        self.temp_dir = tempfile.mkdtemp()
        self.comp_matrix_file = os.path.join(self.temp_dir, "test_comp.csv")
        
        # Create a simple 3x3 compensation matrix
        comp_data = pd.DataFrame(
            [[1.0, 0.1, 0.05],
             [0.05, 1.0, 0.1],
             [0.02, 0.05, 1.0]],
            index=['FITC', 'PE', 'APC'],
            columns=['FITC', 'PE', 'APC']
        )
        comp_data.to_csv(self.comp_matrix_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_compensation_matrix(self):
        """Test loading compensation matrix from CSV."""
        matrix = self.loader.load_from_csv(self.comp_matrix_file)
        
        assert matrix is not None
        assert matrix.shape == (3, 3)
        assert self.loader.markers == ['FITC', 'PE', 'APC']
    
    def test_validate_matrix_square(self):
        """Test matrix validation for square matrix."""
        matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
        markers = ['FITC', 'PE']
        
        assert self.loader.validate_matrix(matrix, markers) is True
    
    def test_validate_matrix_non_square(self):
        """Test matrix validation fails for non-square matrix."""
        matrix = np.array([[1.0, 0.1, 0.05], [0.1, 1.0, 0.05]])
        markers = ['FITC', 'PE']
        
        assert self.loader.validate_matrix(matrix, markers) is False
    
    def test_validate_matrix_dimension_mismatch(self):
        """Test matrix validation fails when dimensions don't match markers."""
        matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
        markers = ['FITC', 'PE', 'APC']
        
        assert self.loader.validate_matrix(matrix, markers) is False
    
    def test_get_marker_names(self):
        """Test extracting marker names."""
        self.loader.load_from_csv(self.comp_matrix_file)
        markers = self.loader.get_marker_names()
        
        assert markers == ['FITC', 'PE', 'APC']
    
    def test_get_marker_names_not_loaded(self):
        """Test getting marker names before loading raises error."""
        with pytest.raises(ValueError):
            self.loader.get_marker_names()
    
    def test_get_matrix(self):
        """Test getting the compensation matrix."""
        self.loader.load_from_csv(self.comp_matrix_file)
        matrix = self.loader.get_matrix()
        
        assert matrix is not None
        assert matrix.shape == (3, 3)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_from_csv("nonexistent.csv")


class TestFCSPreprocessor:
    """Test suite for FCSPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = FCSPreprocessor()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'FITC': [100, 200, 300, 400],
            'PE': [150, 250, 350, 450],
            'APC': [200, 300, 400, 500]
        })
        
        # Create a simple compensation matrix
        self.comp_matrix = np.array([
            [1.0, 0.1, 0.05],
            [0.05, 1.0, 0.1],
            [0.02, 0.05, 1.0]
        ])
        self.markers = ['FITC', 'PE', 'APC']
    
    def test_apply_compensation_basic(self):
        """Test basic compensation application."""
        compensated = self.preprocessor.apply_compensation(
            self.test_data,
            self.comp_matrix,
            self.markers
        )
        
        assert compensated is not None
        assert compensated.shape == self.test_data.shape
        assert list(compensated.columns) == list(self.test_data.columns)
    
    def test_apply_compensation_changes_values(self):
        """Test that compensation actually changes values."""
        compensated = self.preprocessor.apply_compensation(
            self.test_data,
            self.comp_matrix,
            self.markers
        )
        
        # Values should be different after compensation (unless matrix is identity)
        # Since our matrix has off-diagonal elements, values should change
        assert not np.allclose(compensated.values, self.test_data.values)
    
    def test_apply_compensation_marker_mismatch(self):
        """Test compensation with mismatched markers raises error."""
        wrong_markers = ['FITC', 'PE', 'PerCP']  # Wrong marker name
        
        with pytest.raises(ValueError):
            self.preprocessor.apply_compensation(
                self.test_data,
                self.comp_matrix,
                wrong_markers
            )
    
    def test_batch_process_empty_directory(self):
        """Test batch processing with empty directory."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            stats = self.preprocessor.batch_process(
                input_dir=temp_dir,
                comp_matrix=self.comp_matrix,
                markers=self.markers,
                output_dir=temp_dir
            )
            
            assert stats['total_files'] == 0
            assert stats['processed'] == 0
            assert stats['failed'] == 0
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_batch_process_nonexistent_directory(self):
        """Test batch processing with nonexistent directory raises error."""
        with pytest.raises(FileNotFoundError):
            self.preprocessor.batch_process(
                input_dir="/nonexistent/directory",
                comp_matrix=self.comp_matrix,
                markers=self.markers,
                output_dir="/tmp"
            )


class TestMetadataRouter:
    """Test suite for MetadataRouter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = MetadataRouter(base_output_dir="data/")
        
        # Create temporary metadata CSV
        self.temp_dir = tempfile.mkdtemp()
        self.metadata_file = os.path.join(self.temp_dir, "metadata.csv")
        
        metadata = pd.DataFrame({
            'FCSFileName': ['1', '2', '3', '4'],
            'Label': ['HEU', 'UE', 'HEU', 'NA'],
            'Stimulation': ['CPG', 'PIC', 'LPS', 'CPG'],
            'SampleNumber': [1, 2, 1, 3]
        })
        metadata.to_csv(self.metadata_file, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_metadata_csv(self):
        """Test loading metadata from CSV."""
        df = self.router.load_metadata(self.metadata_file)
        
        assert df is not None
        assert len(df) == 4
        assert 'FCSFileName' in df.columns
        assert 'Label' in df.columns
    
    def test_load_metadata_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            self.router.load_metadata("nonexistent.csv")
    
    def test_load_metadata_missing_columns(self):
        """Test loading metadata with missing columns raises error."""
        # Create CSV with missing columns
        bad_file = os.path.join(self.temp_dir, "bad_metadata.csv")
        bad_data = pd.DataFrame({
            'FCSFileName': ['1', '2'],
            'Label': ['HEU', 'UE']
            # Missing Stimulation column
        })
        bad_data.to_csv(bad_file, index=False)
        
        with pytest.raises(ValueError):
            self.router.load_metadata(bad_file)
    
    def test_determine_output_path_heu(self):
        """Test output path determination for HEU samples."""
        path = self.router.determine_output_path(
            filename='1.fcs',
            label='HEU',
            is_training=True,
            stimulation='CPG'
        )
        
        assert 'mixed_training' in path
        assert '1_HEU_CPG.fcs' in path
    
    def test_determine_output_path_ue(self):
        """Test output path determination for UE samples."""
        path = self.router.determine_output_path(
            filename='2.fcs',
            label='UE',
            is_training=True,
            stimulation='PIC'
        )
        
        assert 'mixed_training' in path
        assert '2_UE_PIC.fcs' in path
    
    def test_determine_output_path_na(self):
        """Test output path determination for NA (unlabeled) samples."""
        path = self.router.determine_output_path(
            filename='4.fcs',
            label='NA',
            is_training=False,
            stimulation='CPG'
        )
        
        assert 'mixed' in path
        assert '4_CPG.fcs' in path
    
    def test_determine_output_path_no_stimulation(self):
        """Test output path without stimulation in filename."""
        path = self.router.determine_output_path(
            filename='1.fcs',
            label='HEU',
            is_training=True,
            stimulation='CPG',
            include_stimulation=False
        )
        
        assert 'mixed_training' in path
        assert '1_HEU.fcs' in path
        assert 'CPG' not in path
    
    def test_create_routing_map(self):
        """Test creating routing map from metadata."""
        self.router.load_metadata(self.metadata_file)
        routing_map = self.router.create_routing_map()
        
        assert routing_map is not None
        assert len(routing_map) > 0
        
        # Check that HEU files have both regular and positive entries
        assert '1.fcs' in routing_map
        assert '1.fcs:positive' in routing_map
    
    def test_create_routing_map_not_loaded(self):
        """Test creating routing map before loading metadata raises error."""
        with pytest.raises(ValueError):
            self.router.create_routing_map()
    
    def test_get_routing_statistics(self):
        """Test getting routing statistics."""
        self.router.load_metadata(self.metadata_file)
        self.router.create_routing_map()
        
        stats = self.router.get_routing_statistics()
        
        assert 'positive' in stats
        assert 'mixed_training' in stats
        assert 'mixed' in stats
        assert stats['positive'] > 0  # HEU samples
        assert stats['mixed_training'] > 0  # HEU + UE samples
    
    def test_get_routing_statistics_empty(self):
        """Test getting statistics with empty routing map."""
        stats = self.router.get_routing_statistics()
        assert stats == {}
    
    def test_multiple_metadata_files(self):
        """Test loading multiple metadata files."""
        # Create second metadata file
        metadata2_file = os.path.join(self.temp_dir, "metadata2.csv")
        metadata2 = pd.DataFrame({
            'FCSFileName': ['5', '6'],
            'Label': ['HEU', 'UE'],
            'Stimulation': ['R848', 'PAM'],
            'SampleNumber': [4, 5]
        })
        metadata2.to_csv(metadata2_file, index=False)
        
        # Load both files
        self.router.load_metadata(self.metadata_file)
        self.router.load_metadata(metadata2_file)
        
        assert len(self.router.metadata) == 6  # 4 + 2 entries
    
    def test_metadata_deduplication(self):
        """Test that duplicate entries are handled correctly."""
        # Load same file twice
        self.router.load_metadata(self.metadata_file)
        self.router.load_metadata(self.metadata_file)
        
        # Should have deduplicated (kept last)
        assert len(self.router.metadata) == 4  # Not 8
