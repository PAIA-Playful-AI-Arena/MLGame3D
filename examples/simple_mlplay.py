"""
Simple MLPlay Example

This file demonstrates how to create a minimal MLPlay class for the MLGame3D framework.
"""

import numpy as np
from typing import Dict, Any

class MLPlay:
    """
    A minimal MLPlay class that demonstrates the required methods.
    
    This class provides a simple implementation that returns random actions.
    """
    
    def __init__(self, observation_structure=None, action_space_info=None, name=None, game_params=None):
        """
        Initialize the MLPlay instance.
        
        Args:
            observation_structure: Dictionary describing observation space structure
            action_space_info: ActionSpec object describing action space
            name: Name for this MLPlay instance
            game_params: Dictionary of game parameters
        """
        self.action_space_info = action_space_info
        self.name = name or "SimpleMLPlay"
        self.observation_structure = observation_structure
        self.game_params = game_params or {}
        self.step_counter = 0
        
    def reset(self):
        """
        Reset the agent for a new episode.
        """
        self.step_counter = 0
        
    def update(self, observations, done=False, info=None):
        """
        Process observations and choose an action.
        
        Args:
            observations: Dictionary of observations from the environment
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            Action in the format required by the action space
        """
        self.step_counter += 1
            
        if done:
            print("Episode finished!")
        
        # Simple strategy: alternate between different movement patterns
        if self.action_space_info.is_continuous():
            # Pure continuous action space
            action_size = self.action_space_info.continuous_size
            return np.random.uniform(-1, 1, size=(action_size,))
                
        elif self.action_space_info.is_discrete():
            # Pure discrete action space
            discrete_branches = self.action_space_info.discrete_branches
            return np.array([np.random.randint(0, branch_size) for branch_size in discrete_branches], dtype=np.int32)

        else:
            # Hybrid action space
            action_size = self.action_space_info.continuous_size
            discrete_branches = self.action_space_info.discrete_branches
            continuous = np.random.uniform(-1, 1, size=(action_size,))
            discrete = np.array([np.random.randint(0, branch_size) for branch_size in discrete_branches], dtype=np.int32)
            return (continuous, discrete)
