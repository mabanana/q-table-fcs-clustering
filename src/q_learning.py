"""
Q-Learning Algorithm Module

Implements the Q-learning reinforcement learning algorithm for cluster selection.
"""

import logging
import pickle
from typing import Dict, Tuple, Optional, List
import numpy as np
import random

logger = logging.getLogger(__name__)


class QLearningAgent:
    """
    Q-Learning agent for selecting optimal number of clusters.
    
    Uses epsilon-greedy exploration and standard Q-learning update rule.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            n_states: Number of possible states in the state space
            n_actions: Number of possible actions (cluster counts)
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay factor for epsilon per episode
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Track training statistics
        self.episode_rewards = []
        self.episode_actions = []
        self.exploration_count = 0
        self.exploitation_count = 0
        
        logger.info(
            f"Initialized Q-Learning Agent: "
            f"{n_states} states, {n_actions} actions, "
            f"lr={learning_rate}, gamma={discount_factor}, "
            f"epsilon={epsilon_start}->{epsilon_end}"
        )
    
    def choose_action(self, state: int, explore: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use exploration (epsilon-greedy) or pure exploitation
            
        Returns:
            Selected action index
        """
        if state < 0 or state >= self.n_states:
            logger.warning(f"Invalid state {state}, clipping to valid range")
            state = np.clip(state, 0, self.n_states - 1)
        
        if explore and random.random() < self.epsilon:
            # Exploration: choose random action
            action = random.randint(0, self.n_actions - 1)
            self.exploration_count += 1
            logger.debug(f"Exploration: state={state}, action={action}")
        else:
            # Exploitation: choose best action based on Q-values
            action = np.argmax(self.q_table[state])
            self.exploitation_count += 1
            logger.debug(
                f"Exploitation: state={state}, action={action}, "
                f"Q-value={self.q_table[state, action]:.4f}"
            )
        
        return action
    
    def update_q_value(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int
    ) -> float:
        """
        Update Q-value using the Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            
        Returns:
            TD error (temporal difference error)
        """
        # Validate inputs
        if state < 0 or state >= self.n_states:
            logger.warning(f"Invalid state {state}, clipping to valid range")
            state = np.clip(state, 0, self.n_states - 1)
        
        if next_state < 0 or next_state >= self.n_states:
            logger.warning(f"Invalid next_state {next_state}, clipping to valid range")
            next_state = np.clip(next_state, 0, self.n_states - 1)
        
        if action < 0 or action >= self.n_actions:
            logger.warning(f"Invalid action {action}, clipping to valid range")
            action = np.clip(action, 0, self.n_actions - 1)
        
        # Get current Q-value
        current_q = self.q_table[state, action]
        
        # Get max Q-value for next state
        max_next_q = np.max(self.q_table[next_state])
        
        # Calculate TD target and error
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * td_error
        self.q_table[state, action] = new_q
        
        logger.debug(
            f"Q-update: s={state}, a={action}, r={reward:.4f}, "
            f"Q: {current_q:.4f} -> {new_q:.4f}, TD error: {td_error:.4f}"
        )
        
        return td_error
    
    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon).
        Called at the end of each episode.
        """
        old_epsilon = self.epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        if old_epsilon != self.epsilon:
            logger.debug(f"Epsilon decayed: {old_epsilon:.4f} -> {self.epsilon:.4f}")
    
    def reset_epsilon(self):
        """
        Reset epsilon to its starting value.
        Useful when starting a new training phase.
        """
        self.epsilon = self.epsilon_start
        logger.info(f"Epsilon reset to {self.epsilon}")
    
    def reset_counters(self):
        """
        Reset exploration and exploitation counters.
        Useful for tracking statistics per training phase.
        """
        self.exploration_count = 0
        self.exploitation_count = 0
        logger.info("Exploration/exploitation counters reset")
    
    def record_episode(self, total_reward: float, actions: List[int]):
        """
        Record statistics from an episode.
        
        Args:
            total_reward: Total reward accumulated in the episode
            actions: List of actions taken in the episode
        """
        self.episode_rewards.append(total_reward)
        self.episode_actions.extend(actions)
        
        logger.debug(f"Episode recorded: reward={total_reward:.4f}, actions={len(actions)}")
    
    def get_statistics(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'epsilon': self.epsilon,
            'n_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'total_exploration': self.exploration_count,
            'total_exploitation': self.exploitation_count,
            'exploration_rate': (
                self.exploration_count / (self.exploration_count + self.exploitation_count)
                if (self.exploration_count + self.exploitation_count) > 0 else 0
            )
        }
        
        if self.episode_rewards:
            stats['min_reward'] = np.min(self.episode_rewards)
            stats['max_reward'] = np.max(self.episode_rewards)
            stats['last_10_mean_reward'] = np.mean(self.episode_rewards[-10:])
        
        if self.episode_actions:
            stats['action_distribution'] = {
                action: self.episode_actions.count(action)
                for action in range(self.n_actions)
            }
        
        return stats
    
    def get_policy(self) -> np.ndarray:
        """
        Get the current policy (best action for each state).
        
        Returns:
            Array of shape (n_states,) with best action for each state
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_state_values(self) -> np.ndarray:
        """
        Get the value of each state (max Q-value).
        
        Returns:
            Array of shape (n_states,) with value of each state
        """
        return np.max(self.q_table, axis=1)
    
    def save(self, filepath: str):
        """
        Save the Q-table and agent parameters to a file.
        
        Args:
            filepath: Path to save the pickle file
        """
        state = {
            'q_table': self.q_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'episode_rewards': self.episode_rewards,
            'episode_actions': self.episode_actions,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Q-table saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the Q-table and agent parameters from a file.
        
        Args:
            filepath: Path to the pickle file
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore state
        self.q_table = state['q_table']
        self.n_states = state['n_states']
        self.n_actions = state['n_actions']
        self.learning_rate = state['learning_rate']
        self.discount_factor = state['discount_factor']
        self.epsilon = state['epsilon']
        self.epsilon_start = state['epsilon_start']
        self.epsilon_end = state['epsilon_end']
        self.epsilon_decay = state['epsilon_decay']
        self.episode_rewards = state['episode_rewards']
        self.episode_actions = state['episode_actions']
        self.exploration_count = state['exploration_count']
        self.exploitation_count = state['exploitation_count']
        
        logger.info(f"Q-table loaded from {filepath}")
        logger.info(f"Restored {len(self.episode_rewards)} episodes of training history")
    
    def get_q_table_stats(self) -> Dict:
        """
        Get statistics about the Q-table.
        
        Returns:
            Dictionary with Q-table statistics
        """
        non_zero_q = self.q_table[self.q_table != 0]
        
        stats = {
            'shape': self.q_table.shape,
            'mean_q': np.mean(self.q_table),
            'std_q': np.std(self.q_table),
            'min_q': np.min(self.q_table),
            'max_q': np.max(self.q_table),
            'non_zero_entries': len(non_zero_q),
            'non_zero_percentage': (
                len(non_zero_q) / self.q_table.size * 100
            ),
            'state_coverage': (
                np.sum(np.any(self.q_table != 0, axis=1)) / self.n_states * 100
            )
        }
        
        if len(non_zero_q) > 0:
            stats['mean_non_zero_q'] = np.mean(non_zero_q)
            stats['std_non_zero_q'] = np.std(non_zero_q)
        
        return stats


def create_action_space(min_clusters: int, max_clusters: int) -> Dict[int, int]:
    """
    Create a mapping from action indices to cluster counts.
    
    Args:
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        
    Returns:
        Dictionary mapping action index to cluster count
    """
    action_to_clusters = {
        i: n_clusters
        for i, n_clusters in enumerate(range(min_clusters, max_clusters + 1))
    }
    
    logger.info(f"Created action space: {action_to_clusters}")
    return action_to_clusters


def clusters_to_action(n_clusters: int, action_space: Dict[int, int]) -> int:
    """
    Convert a cluster count to an action index.
    
    Args:
        n_clusters: Number of clusters
        action_space: Action space mapping
        
    Returns:
        Action index
    """
    for action, clusters in action_space.items():
        if clusters == n_clusters:
            return action
    
    raise ValueError(f"Cluster count {n_clusters} not in action space")
