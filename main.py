#!/usr/bin/env python3
"""
Main Execution Script for Q-Learning FCS Clustering System

A reinforcement learning approach to HIV diagnosis using flow cytometry data.
This is a science fair project demonstrating RL applied to medical diagnostics.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

import numpy as np

from src.fcs_loader import FCSLoader
from src.discretizer import ClinicalDiscretizer
from src.q_learning import QLearningAgent, create_action_space
from src.clustering import ClusteringEngine
from src.trainer import ReinforcementClusteringPipeline
from src.visualizer import VisualizationEngine


def configure_logging(log_filepath: str = "output/training.log", verbosity: str = "INFO"):
    """Set up logging configuration."""
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
    log_level = getattr(logging, verbosity.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_configuration(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def initialize_components(config: dict, use_gpu: bool = True):
    """Initialize all system components."""
    logger = logging.getLogger(__name__)
    
    # Extract configuration
    markers_config = config['markers']
    q_config = config['q_learning']
    disc_config = config['discretization']
    
    # Initialize FCS loader
    primary_markers = markers_config.get('use_for_clustering', markers_config['primary'])
    fcs_loader = FCSLoader(markers=primary_markers)
    logger.info(f"FCS Loader initialized with markers: {primary_markers}")
    
    # Initialize discretizer
    discretizer = ClinicalDiscretizer(
        cd4_bins=disc_config['cd4_bins'],
        cd4_cd8_ratio_bins=disc_config['cd4_cd8_ratio_bins'],
        activation_bins=disc_config['activation_bins']
    )
    logger.info("Clinical Discretizer initialized")
    logger.info("\n" + discretizer.explain_discretization())
    
    # Initialize clustering engine
    clustering_engine = ClusteringEngine(
        use_gpu=use_gpu,
        fallback_to_cpu=config['gpu']['fallback_to_cpu']
    )
    
    # Create action space
    min_clusters, max_clusters = q_config['n_clusters_range']
    action_space = create_action_space(min_clusters, max_clusters)
    n_actions = len(action_space)
    
    # Determine number of states (2 or 3 features, 4 bins each)
    use_activation = 'HLA-DR' in markers_config.get('secondary', [])
    n_states = 64 if use_activation else 16
    
    # Initialize Q-learning agent
    q_agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=q_config['learning_rate'],
        discount_factor=q_config['discount_factor'],
        epsilon_start=q_config['epsilon_start'],
        epsilon_end=q_config['epsilon_end'],
        epsilon_decay=q_config['epsilon_decay']
    )
    logger.info(f"Q-Learning Agent initialized: {n_states} states, {n_actions} actions")
    
    # Initialize trainer
    trainer = ReinforcementClusteringPipeline(
        q_agent=q_agent,
        action_space=action_space,
        discretizer=discretizer,
        fcs_loader=fcs_loader,
        clustering_engine=clustering_engine,
        use_activation=use_activation
    )
    
    return trainer, q_agent, action_space, discretizer


def execute_phase1_training(trainer, config: dict, episodes: int = None):
    """Execute Phase 1: Quality-based learning on positive samples."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 1: QUALITY-BASED LEARNING")
    logger.info("=" * 80)
    
    positive_data_dir = config['data']['positive']
    n_episodes = episodes or config['training']['stage1_episodes']
    batch_size = config['training']['batch_size']
    
    logger.info(f"Training for {n_episodes} episodes...")
    
    history = trainer.execute_quality_learning_phase(
        positive_samples_path=positive_data_dir,
        learning_iterations=n_episodes,
        samples_per_iteration=batch_size,
        quality_measure="silhouette"
    )
    
    logger.info("Phase 1 training completed successfully")
    return history


def execute_phase2_training(trainer, config: dict, episodes: int = None):
    """Execute Phase 2: Diagnostic refinement on labeled mixed samples."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PHASE 2: DIAGNOSTIC REFINEMENT")
    logger.info("=" * 80)
    
    mixed_training_dir = config['data']['mixed_training']
    n_episodes = episodes or config['training']['stage2_episodes']
    batch_size = config['training']['batch_size']
    
    logger.info(f"Training for {n_episodes} episodes...")
    
    history = trainer.execute_diagnostic_refinement_phase(
        labeled_samples_path=mixed_training_dir,
        learning_iterations=n_episodes,
        samples_per_iteration=batch_size,
        accuracy_measure="f1"
    )
    
    logger.info("Phase 2 training completed successfully")
    return history


def execute_testing(trainer, config: dict, input_dir: str = None, output_file: str = None):
    """Execute testing/inference on unlabeled data."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("INFERENCE MODE")
    logger.info("=" * 80)
    
    test_data_dir = input_dir or config['data']['mixed_test']
    output_path = output_file or config['output']['results_path']
    
    logger.info(f"Applying learned policy to: {test_data_dir}")
    
    results = trainer.apply_learned_policy(
        test_samples_path=test_data_dir,
        predictions_output=output_path
    )
    
    logger.info(f"Predictions saved to: {output_path}")
    logger.info(f"Total predictions: {len(results)}")
    
    # Print summary statistics
    if 'predicted_hiv_status' in results.columns:
        positive_count = (results['predicted_hiv_status'] == 'positive').sum()
        negative_count = (results['predicted_hiv_status'] == 'negative').sum()
        logger.info(f"Predicted positive: {positive_count}, negative: {negative_count}")
    
    return results


def execute_visualization(q_agent, trainer, action_space, discretizer, config: dict):
    """Generate all visualizations."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    viz_config = config['visualization']
    viz_engine = VisualizationEngine(
        output_directory=viz_config['output_dir'],
        dpi=viz_config['dpi'],
        figure_dimensions=tuple(viz_config['figure_size']),
        color_palette=viz_config['color_scheme']
    )
    
    # Plot learning curves
    if trainer.phase1_progress:
        logger.info("Creating Phase 1 learning curve...")
        viz_engine.plot_learning_curve(
            trainer.phase1_progress,
            "Phase 1: Quality Learning",
            save_filename="phase1_learning_curve.png"
        )
    
    if trainer.phase2_progress:
        logger.info("Creating Phase 2 learning curve...")
        viz_engine.plot_learning_curve(
            trainer.phase2_progress,
            "Phase 2: Diagnostic Refinement",
            save_filename="phase2_learning_curve.png"
        )
    
    # Plot Q-value heatmap
    logger.info("Creating Q-value heatmap...")
    action_labels = [f"k={action_space[i]}" for i in range(len(action_space))]
    viz_engine.plot_q_value_heatmap(
        q_agent.q_table,
        action_labels,
        save_filename="q_value_heatmap.png"
    )
    
    # Plot action distribution
    if q_agent.episode_actions:
        logger.info("Creating action distribution plot...")
        action_counts = {}
        for action in q_agent.episode_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        action_label_map = {i: f"k={action_space[i]}" for i in range(len(action_space))}
        viz_engine.plot_action_distribution(
            action_counts,
            action_label_map,
            save_filename="action_distribution.png"
        )
    
    # Generate summary report
    if trainer.phase1_progress and trainer.phase2_progress:
        logger.info("Generating summary report...")
        q_stats = q_agent.get_q_table_stats()
        viz_engine.generate_summary_report(
            trainer.phase1_progress,
            trainer.phase2_progress,
            q_stats,
            output_filename="training_summary.html"
        )
    
    logger.info("Visualization generation completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Q-Learning FCS Clustering System for HIV Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both phases
  python main.py --train-full --episodes 2000
  
  # Train only Phase 1
  python main.py --train-phase1 --episodes 1000
  
  # Test on unlabeled data
  python main.py --test --input data/mixed/ --output output/predictions.csv
  
  # Generate visualizations
  python main.py --visualize --q-table output/q_table.pkl
        """
    )
    
    # Mode selection
    mode_group = parser.add_argument_group('Execution Modes')
    mode_group.add_argument('--train-phase1', action='store_true',
                           help='Train Phase 1 only (quality learning on HIV+ data)')
    mode_group.add_argument('--train-phase2', action='store_true',
                           help='Train Phase 2 only (diagnostic refinement on labeled data)')
    mode_group.add_argument('--train-full', action='store_true',
                           help='Train both phases sequentially')
    mode_group.add_argument('--test', action='store_true',
                           help='Run inference on unlabeled data')
    mode_group.add_argument('--visualize', action='store_true',
                           help='Generate all visualizations')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str, default='config.yaml',
                             help='Path to configuration file (default: config.yaml)')
    config_group.add_argument('--episodes', type=int,
                             help='Number of training episodes (overrides config)')
    config_group.add_argument('--use-gpu', action='store_true', default=False,
                             help='Attempt to use GPU acceleration')
    config_group.add_argument('--q-table', type=str, default='output/q_table.pkl',
                             help='Path to Q-table file for save/load')
    
    # I/O options
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', type=str,
                         help='Input directory for test mode')
    io_group.add_argument('--output', type=str,
                         help='Output file for predictions')
    io_group.add_argument('--log-level', type=str, default='INFO',
                         choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                         help='Logging verbosity level')
    
    args = parser.parse_args()
    
    # Configure logging
    logger = configure_logging(verbosity=args.log_level)
    
    logger.info("=" * 80)
    logger.info("Q-LEARNING FCS CLUSTERING SYSTEM")
    logger.info("Reinforcement Learning for HIV Diagnosis")
    logger.info("=" * 80)
    
    # Load configuration
    try:
        config = load_configuration(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    # Initialize components
    try:
        trainer, q_agent, action_space, discretizer = initialize_components(
            config,
            use_gpu=args.use_gpu
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        sys.exit(1)
    
    # Execute requested mode
    try:
        if args.train_full:
            # Full training: both phases
            logger.info("Starting full training (Phase 1 + Phase 2)")
            
            # Phase 1
            phase1_history = execute_phase1_training(trainer, config, args.episodes)
            
            # Save intermediate Q-table
            intermediate_path = args.q_table.replace('.pkl', '_phase1.pkl')
            q_agent.save(intermediate_path)
            logger.info(f"Phase 1 Q-table saved to: {intermediate_path}")
            
            # Phase 2
            phase2_history = execute_phase2_training(trainer, config, args.episodes)
            
            # Save final Q-table
            q_agent.save(args.q_table)
            logger.info(f"Final Q-table saved to: {args.q_table}")
            
            # Generate visualizations
            execute_visualization(q_agent, trainer, action_space, discretizer, config)
            
        elif args.train_phase1:
            # Phase 1 only
            phase1_history = execute_phase1_training(trainer, config, args.episodes)
            q_agent.save(args.q_table)
            logger.info(f"Phase 1 Q-table saved to: {args.q_table}")
            execute_visualization(q_agent, trainer, action_space, discretizer, config)
            
        elif args.train_phase2:
            # Phase 2 only (requires existing Q-table)
            if os.path.exists(args.q_table):
                logger.info(f"Loading existing Q-table from: {args.q_table}")
                q_agent.load(args.q_table)
            else:
                logger.warning(f"Q-table not found at {args.q_table}, starting fresh")
            
            phase2_history = execute_phase2_training(trainer, config, args.episodes)
            q_agent.save(args.q_table)
            logger.info(f"Phase 2 Q-table saved to: {args.q_table}")
            execute_visualization(q_agent, trainer, action_space, discretizer, config)
            
        elif args.test:
            # Testing/inference mode
            if not os.path.exists(args.q_table):
                logger.error(f"Q-table not found at {args.q_table}. Train first!")
                sys.exit(1)
            
            logger.info(f"Loading Q-table from: {args.q_table}")
            q_agent.load(args.q_table)
            
            results = execute_testing(trainer, config, args.input, args.output)
            
        elif args.visualize:
            # Visualization only
            if not os.path.exists(args.q_table):
                logger.error(f"Q-table not found at {args.q_table}")
                sys.exit(1)
            
            logger.info(f"Loading Q-table from: {args.q_table}")
            q_agent.load(args.q_table)
            
            execute_visualization(q_agent, trainer, action_space, discretizer, config)
            
        else:
            logger.error("No execution mode specified. Use --help for usage information.")
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
