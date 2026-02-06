#!/usr/bin/env python3
"""
FCS Data Preprocessing Script

Applies compensation matrices and routes files based on metadata.

Usage:
    python preprocess_data.py --raw-dir data/raw/ --compensation data/compensation/0001.csv
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will not be shown.")

from src.preprocessing import (
    CompensationMatrixLoader,
    FCSPreprocessor,
    MetadataRouter
)


def setup_logging(log_file: str = "output/preprocessing.log", verbose: bool = False):
    """Set up logging configuration."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('preprocessing', {})
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


def print_summary(stats: Dict[str, Any], routing_stats: Dict[str, int]):
    """Print processing summary."""
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"\nProcessing Statistics:")
    print(f"  Total files found:      {stats.get('total_files', 0)}")
    print(f"  Successfully processed: {stats.get('processed', 0)}")
    print(f"  Failed:                 {stats.get('failed', 0)}")
    print(f"  Skipped (existing):     {stats.get('skipped', 0)}")
    
    print(f"\nFile Routing:")
    print(f"  /data/positive/:        {routing_stats.get('positive', 0)}")
    print(f"  /data/mixed_training/:  {routing_stats.get('mixed_training', 0)}")
    print(f"  /data/mixed/:           {routing_stats.get('mixed', 0)}")
    print(f"  Other:                  {routing_stats.get('other', 0)}")
    
    print("\n" + "="*60)


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='Preprocess raw FCS data with compensation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with default settings
  python preprocess_data.py
  
  # Custom paths and multiple metadata files
  python preprocess_data.py \\
    --raw-dir data/raw/ \\
    --compensation data/compensation/0001.csv \\
    --metadata data/compensation/HEUvsUE.csv data/compensation/HEUvsUETraining.csv
  
  # Dry run to preview operations
  python preprocess_data.py --dry-run
  
  # Verbose logging
  python preprocess_data.py --verbose
        """
    )
    
    parser.add_argument(
        '--raw-dir',
        default='data/raw/',
        help='Directory with raw FCS files (default: data/raw/)'
    )
    parser.add_argument(
        '--compensation',
        default='data/compensation/0001.csv',
        help='Compensation matrix CSV (default: data/compensation/0001.csv)'
    )
    parser.add_argument(
        '--metadata',
        nargs='+',
        default=['data/compensation/HEUvsUE.csv'],
        help='Metadata CSV file(s) (default: data/compensation/HEUvsUE.csv)'
    )
    parser.add_argument(
        '--output-base',
        default='data/',
        help='Base output directory (default: data/)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without processing'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-stimulation',
        action='store_true',
        help='Do not include stimulation in output filenames'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    logger.info("Starting FCS preprocessing pipeline")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override with command-line arguments
    raw_dir = args.raw_dir
    comp_file = args.compensation
    metadata_files = args.metadata
    output_base = args.output_base
    include_stimulation = not args.no_stimulation
    
    # Print configuration
    print("\n" + "="*60)
    print("FCS PREPROCESSING PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Raw data directory:    {raw_dir}")
    print(f"  Compensation matrix:   {comp_file}")
    print(f"  Metadata files:        {metadata_files}")
    print(f"  Output base directory: {output_base}")
    print(f"  Include stimulation:   {include_stimulation}")
    print(f"  Dry run:               {args.dry_run}")
    print(f"  Overwrite existing:    {args.overwrite}")
    print("="*60 + "\n")
    
    try:
        # Step 1: Load compensation matrix
        logger.info("Loading compensation matrix...")
        comp_loader = CompensationMatrixLoader()
        comp_matrix = comp_loader.load_from_csv(comp_file)
        markers = comp_loader.get_marker_names()
        
        print(f"✓ Loaded compensation matrix: {comp_matrix.shape}")
        print(f"  Markers: {', '.join(markers[:5])}{'...' if len(markers) > 5 else ''}")
        print()
        
        # Step 2: Load metadata
        logger.info("Loading metadata...")
        router = MetadataRouter(base_output_dir=output_base)
        
        for metadata_file in metadata_files:
            try:
                router.load_metadata(metadata_file)
                print(f"✓ Loaded metadata from: {metadata_file}")
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")
                print(f"✗ Failed to load: {metadata_file}")
        
        print()
        
        # Step 3: Create routing map
        logger.info("Creating routing map...")
        routing_map = router.create_routing_map(
            include_stimulation=include_stimulation,
            positive_label=config.get('routing', {}).get('positive_label', 'HEU')
        )
        
        routing_stats = router.get_routing_statistics()
        print(f"✓ Created routing map: {len(routing_map)} file routes")
        print(f"  Positive samples:       {routing_stats.get('positive', 0)}")
        print(f"  Mixed training samples: {routing_stats.get('mixed_training', 0)}")
        print(f"  Test samples:           {routing_stats.get('mixed', 0)}")
        print()
        
        # Step 4: Process files
        if args.dry_run:
            print("DRY RUN MODE - No files will be processed")
            print("\nSample routing (first 10 entries):")
            for i, (input_file, output_path) in enumerate(list(routing_map.items())[:10]):
                print(f"  {input_file} -> {output_path}")
            print()
            logger.info("Dry run complete")
            return
        
        # Initialize preprocessor
        logger.info("Initializing FCS preprocessor...")
        preprocessor = FCSPreprocessor()
        
        # Process files
        print("Processing FCS files...")
        logger.info(f"Starting batch processing of {raw_dir}")
        
        stats = preprocessor.batch_process(
            input_dir=raw_dir,
            comp_matrix=comp_matrix,
            markers=markers,
            output_dir=output_base,
            routing_map=routing_map
        )
        
        # Print summary
        print_summary(stats, routing_stats)
        
        # Log completion
        logger.info("Preprocessing pipeline completed successfully")
        
        if stats.get('failed', 0) > 0:
            print(f"\n⚠ Warning: {stats['failed']} files failed to process")
            print(f"  Check log file for details: output/preprocessing.log")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        print(f"  Check log file for details: output/preprocessing.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
