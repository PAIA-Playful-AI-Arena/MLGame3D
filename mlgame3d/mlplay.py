"""
MLPlay Module

This module provides a base class for implementing MLPlay classes.
"""

import numpy as np
from typing import Dict, Any

class RandomMLPlay:
    """
    A class that takes random actions.
    """
    
    def __init__(self, action_space_info, name: str = "RandomMLPlay", parameters: Dict[str, Any] = None):
        """
        Initialize the random MLPlay instance.
        
        Args:
            action_space_info: Information about the action space
            name: The name of the MLPlay instance
            parameters: Optional dictionary of game parameters
        """
        self.action_space_info = action_space_info
        self.name = name
        self.parameters = parameters or {}
    
    def reset(self):
        """
        Reset the MLPlay instance for a new episode.
        """
        pass
    
    def update(self, 
               observations: Dict[str, np.ndarray], 
               done: bool = False, 
               info: Dict[str, Any] = None) -> np.ndarray:
        """
        Process observations and choose a random action.
        
        Args:
            observations: A dictionary of observations
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            A random action
        """
        # Choose a random action
        if self.action_space_info.is_continuous():
            # For continuous action spaces, return random values between -1 and 1
            return np.random.uniform(-1, 1, self.action_space_info.continuous_size)
        elif self.action_space_info.is_discrete():
            # For discrete action spaces, return random integers for each branch
            return np.array([
                np.random.randint(0, branch) 
                for branch in self.action_space_info.discrete_branches
            ], dtype=np.int32)
        else:
            # For hybrid action spaces, return both continuous and discrete actions
            continuous = np.random.uniform(-1, 1, self.action_space_info.continuous_size)
            discrete = np.array([
                np.random.randint(0, branch) 
                for branch in self.action_space_info.discrete_branches
            ], dtype=np.int32)
            return (continuous, discrete)
