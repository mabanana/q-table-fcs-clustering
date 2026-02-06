"""
Unit tests for the Q-Learning module.
"""

import pytest
import numpy as np
import tempfile
import os
from src.q_learning import QLearningAgent, create_action_space, clusters_to_action


class TestQLearningAgent:
    """Test suite for QLearningAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_states = 16
        self.n_actions = 9  # 2-10 clusters
        self.agent = QLearningAgent(
            n_states=self.n_states,
            n_actions=self.n_actions,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon_start=0.3,
            epsilon_end=0.05,
            epsilon_decay=0.995
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.n_states == self.n_states
        assert self.agent.n_actions == self.n_actions
        assert self.agent.learning_rate == 0.1
        assert self.agent.discount_factor == 0.9
        assert self.agent.epsilon == 0.3
        assert self.agent.q_table.shape == (self.n_states, self.n_actions)
        assert np.all(self.agent.q_table == 0)
    
    def test_choose_action_exploration(self):
        """Test action selection with exploration."""
        np.random.seed(42)
        
        # With random Q-values, should explore
        actions = []
        for _ in range(100):
            action = self.agent.choose_action(state=0, explore=True)
            actions.append(action)
        
        # Should have variety due to exploration
        unique_actions = len(set(actions))
        assert unique_actions > 1
    
    def test_choose_action_exploitation(self):
        """Test action selection without exploration."""
        # Set specific Q-values
        self.agent.q_table[0, 3] = 1.0  # Make action 3 best for state 0
        
        # Without exploration, should always choose best action
        actions = []
        for _ in range(10):
            action = self.agent.choose_action(state=0, explore=False)
            actions.append(action)
        
        assert all(a == 3 for a in actions)
    
    def test_update_q_value(self):
        """Test Q-value update."""
        state = 0
        action = 2
        reward = 1.0
        next_state = 1
        
        # Initial Q-value should be 0
        initial_q = self.agent.q_table[state, action]
        assert initial_q == 0
        
        # Update Q-value
        td_error = self.agent.update_q_value(state, action, reward, next_state)
        
        # Q-value should have increased
        new_q = self.agent.q_table[state, action]
        assert new_q > initial_q
        assert new_q == pytest.approx(0.1, abs=0.01)  # lr * (reward + gamma * 0 - 0)
    
    def test_update_q_value_formula(self):
        """Test Q-learning update formula correctness."""
        state = 0
        action = 1
        reward = 2.0
        next_state = 2
        
        # Set up some Q-values
        self.agent.q_table[state, action] = 0.5
        self.agent.q_table[next_state, :] = [0.3, 0.7, 0.4, 0.2, 0.1, 0.5, 0.6, 0.3, 0.2]
        
        max_next_q = 0.7
        expected_td_target = reward + self.agent.discount_factor * max_next_q
        expected_new_q = 0.5 + self.agent.learning_rate * (expected_td_target - 0.5)
        
        # Update
        td_error = self.agent.update_q_value(state, action, reward, next_state)
        
        # Check result
        actual_new_q = self.agent.q_table[state, action]
        assert actual_new_q == pytest.approx(expected_new_q, abs=1e-6)
    
    def test_decay_epsilon(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        self.agent.decay_epsilon()
        
        # Epsilon should have decreased
        assert self.agent.epsilon < initial_epsilon
        assert self.agent.epsilon == pytest.approx(initial_epsilon * 0.995, abs=1e-6)
    
    def test_epsilon_minimum(self):
        """Test that epsilon respects minimum value."""
        # Decay many times
        for _ in range(1000):
            self.agent.decay_epsilon()
        
        # Should not go below epsilon_end
        assert self.agent.epsilon >= self.agent.epsilon_end
        assert self.agent.epsilon == pytest.approx(0.05, abs=1e-6)
    
    def test_reset_epsilon(self):
        """Test epsilon reset."""
        # Decay epsilon
        for _ in range(10):
            self.agent.decay_epsilon()
        
        assert self.agent.epsilon < 0.3
        
        # Reset
        self.agent.reset_epsilon()
        
        assert self.agent.epsilon == 0.3
    
    def test_record_episode(self):
        """Test episode recording."""
        reward = 5.5
        actions = [1, 2, 3, 1, 2]
        
        self.agent.record_episode(reward, actions)
        
        assert len(self.agent.episode_rewards) == 1
        assert self.agent.episode_rewards[0] == reward
        assert len(self.agent.episode_actions) == 5
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        # Record some episodes
        self.agent.record_episode(1.0, [1, 2])
        self.agent.record_episode(2.0, [1, 3])
        self.agent.record_episode(1.5, [2, 2])
        
        stats = self.agent.get_statistics()
        
        assert stats['n_episodes'] == 3
        assert stats['mean_reward'] == pytest.approx(1.5, abs=0.01)
        assert 'action_distribution' in stats
    
    def test_get_policy(self):
        """Test policy extraction."""
        # Set some Q-values
        self.agent.q_table[0, 2] = 1.0  # Best action for state 0 is 2
        self.agent.q_table[1, 5] = 0.8  # Best action for state 1 is 5
        
        policy = self.agent.get_policy()
        
        assert len(policy) == self.n_states
        assert policy[0] == 2
        assert policy[1] == 5
    
    def test_get_state_values(self):
        """Test state value extraction."""
        # Set some Q-values
        self.agent.q_table[0, :] = [0.1, 0.5, 0.3, 0.2, 0.4, 0.1, 0.2, 0.3, 0.1]
        self.agent.q_table[1, :] = [0.8, 0.2, 0.3, 0.1, 0.2, 0.4, 0.5, 0.3, 0.2]
        
        values = self.agent.get_state_values()
        
        assert len(values) == self.n_states
        assert values[0] == pytest.approx(0.5, abs=1e-6)  # Max of state 0
        assert values[1] == pytest.approx(0.8, abs=1e-6)  # Max of state 1
    
    def test_save_and_load(self):
        """Test Q-table save and load."""
        # Set some Q-values and statistics
        self.agent.q_table[0, 2] = 1.5
        self.agent.q_table[1, 3] = 2.0
        self.agent.record_episode(10.0, [1, 2, 3])
        self.agent.epsilon = 0.2
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            self.agent.save(temp_path)
            
            # Create new agent and load
            new_agent = QLearningAgent(
                n_states=self.n_states,
                n_actions=self.n_actions
            )
            new_agent.load(temp_path)
            
            # Check Q-table
            np.testing.assert_array_equal(new_agent.q_table, self.agent.q_table)
            
            # Check parameters
            assert new_agent.epsilon == self.agent.epsilon
            assert new_agent.learning_rate == self.agent.learning_rate
            assert len(new_agent.episode_rewards) == 1
            assert new_agent.episode_rewards[0] == 10.0
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_get_q_table_stats(self):
        """Test Q-table statistics."""
        # Set some values
        self.agent.q_table[0, :] = [1.0, 2.0, 0, 0, 0, 0, 0, 0, 0]
        self.agent.q_table[1, :] = [0.5, 0, 0, 0, 0, 0, 0, 0, 0]
        
        stats = self.agent.get_q_table_stats()
        
        assert 'mean_q' in stats
        assert 'max_q' in stats
        assert 'non_zero_entries' in stats
        assert stats['non_zero_entries'] == 3
        assert stats['max_q'] == 2.0
    
    def test_invalid_state_handling(self):
        """Test handling of invalid state indices."""
        # Negative state
        action = self.agent.choose_action(state=-1, explore=False)
        assert 0 <= action < self.n_actions
        
        # State too large
        action = self.agent.choose_action(state=1000, explore=False)
        assert 0 <= action < self.n_actions
    
    def test_invalid_action_update(self):
        """Test Q-value update with invalid action."""
        # Should handle gracefully by clipping
        td_error = self.agent.update_q_value(
            state=0,
            action=100,  # Invalid action
            reward=1.0,
            next_state=1
        )
        
        # Should not raise an error
        assert td_error is not None


class TestActionSpaceUtils:
    """Test utility functions for action space."""
    
    def test_create_action_space(self):
        """Test action space creation."""
        action_space = create_action_space(2, 10)
        
        assert len(action_space) == 9  # 2,3,4,5,6,7,8,9,10
        assert action_space[0] == 2
        assert action_space[8] == 10
    
    def test_clusters_to_action(self):
        """Test converting cluster count to action index."""
        action_space = create_action_space(2, 10)
        
        action = clusters_to_action(5, action_space)
        assert action == 3  # 5 is at index 3 (2,3,4,5)
    
    def test_clusters_to_action_invalid(self):
        """Test error handling for invalid cluster count."""
        action_space = create_action_space(2, 10)
        
        with pytest.raises(ValueError):
            clusters_to_action(15, action_space)  # 15 not in range


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
