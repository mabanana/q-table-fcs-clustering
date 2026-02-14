"""
FCS File Loader Module

Handles loading and preprocessing of Flow Cytometry Standard (FCS) files.
Uses the fcsparser library to read .fcs files and extract relevant markers.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

try:
    import fcsparser
    FCS_AVAILABLE = True
except ImportError:
    FCS_AVAILABLE = False
    logging.warning("fcsparser not available. FCS file loading will not work.")


logger = logging.getLogger(__name__)


class FCSLoader:
    """
    Handles loading and batch processing of FCS files.
    
    Attributes:
        markers: List of marker names to extract from FCS files
        compensation: Whether to apply compensation matrix if available
        transform: Whether to apply transformation (e.g., log transform)
    """
    
    def __init__(
        self,
        markers: Optional[List[str]] = None,
        compensation: bool = False,
        transform: bool = True
    ):
        """
        Initialize the FCS loader.
        
        Args:
            markers: List of marker names to extract. If None, extracts all markers.
            compensation: Whether to apply compensation if available in FCS file
            transform: Whether to apply log transformation to data
        """
        if not FCS_AVAILABLE:
            raise ImportError(
                "fcsparser is required for FCS file loading. "
                "Install it with: pip install fcsparser"
            )
        
        self.markers = markers
        self.compensation = compensation
        self.transform = transform
        logger.info(f"Initialized FCSLoader with markers: {markers}")
    
    def load_fcs_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load a single FCS file.
        
        Args:
            file_path: Path to the FCS file
            
        Returns:
            Tuple of (data DataFrame, metadata dictionary)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or incompatible
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FCS file not found: {file_path}")
        
        try:
            # Parse the FCS file
            meta, data = fcsparser.parse(
                file_path,
                meta_data_only=False,
                compensate=self.compensation
            )
            
            logger.debug(f"Loaded FCS file: {file_path}")
            logger.debug(f"Available channels: {list(data.columns)}")
            
            # Extract only requested markers if specified
            if self.markers:
                available_markers = self._find_marker_columns(data.columns, self.markers)
                if not available_markers:
                    raise ValueError(
                        f"None of the requested markers {self.markers} found in {file_path}"
                    )
                data = data[available_markers]
                logger.debug(f"Extracted markers: {available_markers}")
            
            # Apply transformation if requested
            if self.transform:
                data = self._apply_transform(data)

            data = self._clean_data_values(data)
            
            return data, meta
            
        except Exception as e:
            logger.error(f"Error loading FCS file {file_path}: {str(e)}")
            raise ValueError(f"Failed to load FCS file {file_path}: {str(e)}")
    
    def load_directory(
        self,
        directory: str,
        pattern: str = "*.fcs"
    ) -> List[Tuple[str, pd.DataFrame, Dict]]:
        """
        Load all FCS files from a directory.
        
        Args:
            directory: Path to directory containing FCS files
            pattern: File pattern to match (default: "*.fcs")
            
        Returns:
            List of tuples: (filename, data DataFrame, metadata dict)
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all FCS files
        fcs_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.fcs'):
                    fcs_files.append(os.path.join(root, file))
        
        if not fcs_files:
            logger.warning(f"No FCS files found in {directory}")
            return []
        
        logger.info(f"Found {len(fcs_files)} FCS files in {directory}")
        
        # Load all files
        results = []
        failed_files = []
        
        for file_path in fcs_files:
            try:
                data, meta = self.load_fcs_file(file_path)
                filename = os.path.basename(file_path)
                results.append((filename, data, meta))
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                failed_files.append(file_path)
        
        if failed_files:
            logger.warning(
                f"Failed to load {len(failed_files)}/{len(fcs_files)} files"
            )
        
        logger.info(f"Successfully loaded {len(results)} FCS files")
        return results
    
    def extract_label_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract label from filename (for labeled training data).
        
        Expected format: patient001_positive.fcs or patient002_negative.fcs
        
        Args:
            filename: Name of the FCS file
            
        Returns:
            Label string ('positive' or 'negative') or None if not found
        """
        filename_lower = filename.lower()
        
        if 'positive' in filename_lower or 'pos' in filename_lower:
            return 'positive'
        elif 'negative' in filename_lower or 'neg' in filename_lower:
            return 'negative'
        
        return None
    
    def _find_marker_columns(
        self,
        columns: pd.Index,
        markers: List[str]
    ) -> List[str]:
        """
        Find column names that match the requested markers.
        
        FCS files may have different naming conventions (e.g., "IFNa", "IFNa-PE", "IFNa PE").
        This method tries to find the best match.
        
        Args:
            columns: Available column names in the FCS file
            markers: Requested marker names
            
        Returns:
            List of matching column names
        """
        matched_columns = []
        
        for marker in markers:
            marker_lower = marker.lower()
            
            # Try exact match first
            if marker in columns:
                matched_columns.append(marker)
                continue
            
            # Try case-insensitive match
            for col in columns:
                if marker_lower == col.lower():
                    matched_columns.append(col)
                    break
            else:
                # Try partial match (e.g., IFNa matches IFNa-PE)
                for col in columns:
                    if marker_lower in col.lower():
                        matched_columns.append(col)
                        logger.debug(f"Matched marker '{marker}' to column '{col}'")
                        break
        
        return matched_columns
    
    def _apply_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to the data.
        
        Uses a log-like transformation that's common in flow cytometry:
        asinh(x / 150) which is similar to log but handles negative values.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Transformed data DataFrame
        """
        # Use asinh transformation (inverse hyperbolic sine)
        # This is standard in flow cytometry as it handles negative values
        # and approximates log for positive values
        transformed = data.copy()
        
        for col in transformed.columns:
            # Apply asinh with cofactor of 150 (standard for flow cytometry)
            transformed[col] = np.arcsinh(transformed[col] / 150.0)
        
        logger.debug("Applied asinh transformation to data")
        return transformed

    def _clean_data_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NaN/Inf values in data.

        Strategy:
        - Replace inf/-inf with NaN
        - Fill NaNs with column medians
        - If any NaNs remain (all-NaN columns), fill with 0
        """
        cleaned = data.replace([np.inf, -np.inf], np.nan)

        if cleaned.isna().any().any():
            medians = cleaned.median(numeric_only=True)
            cleaned = cleaned.fillna(medians)

        if cleaned.isna().any().any():
            cleaned = cleaned.fillna(0.0)

        return cleaned
    
    def aggregate_data(
        self,
        file_results: List[Tuple[str, pd.DataFrame, Dict]],
        aggregation: str = "median"
    ) -> pd.DataFrame:
        """
        Aggregate data from multiple FCS files into summary statistics.
        
        Args:
            file_results: List of (filename, data, metadata) tuples
            aggregation: Aggregation method ('median', 'mean', or 'all')
            
        Returns:
            DataFrame with aggregated values per file
        """
        if not file_results:
            return pd.DataFrame()
        
        aggregated_data = []
        
        for filename, data, meta in file_results:
            if aggregation == "median":
                summary = data.median()
            elif aggregation == "mean":
                summary = data.mean()
            elif aggregation == "all":
                # For 'all', we return the full dataset
                # Add a column for the filename
                data_copy = data.copy()
                data_copy['filename'] = filename
                aggregated_data.append(data_copy)
                continue
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            summary_dict = summary.to_dict()
            summary_dict['filename'] = filename
            
            # Extract label if available
            label = self.extract_label_from_filename(filename)
            if label:
                summary_dict['label'] = label
            
            aggregated_data.append(summary_dict)
        
        if aggregation == "all":
            # Concatenate all data
            result = pd.concat(aggregated_data, ignore_index=True)
        else:
            # Create DataFrame from summary dictionaries
            result = pd.DataFrame(aggregated_data)
        
        logger.info(f"Aggregated data from {len(file_results)} files using {aggregation}")
        return result


def load_fcs_batch(
    directory: str,
    markers: Optional[List[str]] = None,
    aggregation: str = "median"
) -> pd.DataFrame:
    """
    Convenience function to load and aggregate FCS files from a directory.
    
    Args:
        directory: Path to directory containing FCS files
        markers: List of marker names to extract
        aggregation: Aggregation method ('median', 'mean', or 'all')
        
    Returns:
        DataFrame with aggregated data
    """
    loader = FCSLoader(markers=markers)
    file_results = loader.load_directory(directory)
    return loader.aggregate_data(file_results, aggregation=aggregation)
