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
    
    def __init__(self, action_space_info=None):
        """
        Initialize the MLPlay instance.
        
        Args:
            action_space_info: Information about the action space (optional)
        """
        self.action_space_info = action_space_info
        
    def reset(self):
        """
        Reset the agent for a new episode.
        """
        pass
        
    def update(self, 
               observations: Dict[str, np.ndarray], 
               reward: float = 0.0, 
               done: bool = False, 
               info: Dict[str, Any] = None) -> np.ndarray:
        """
        Process observations and choose an action.
        
        This simple implementation just returns a random 2D movement vector.
        
        Args:
            observations: A dictionary of observations
            reward: The reward received
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            The action to take (random 2D movement vector)
        """
        # Print some basic information
        if "obs_0" in observations:
            print(f"Observation shape: {observations['obs_0'].shape}")
            
        if reward != 0:
            print(f"Received reward: {reward:.2f}")
            
        if done:
            print("Episode finished!")
        
        # Return a random 2D movement vector
        action = np.random.uniform(-1, 1, 2)
        
        # Normalize the action vector
        if np.linalg.norm(action) > 0:
            action = action / np.linalg.norm(action)
            
        return action
