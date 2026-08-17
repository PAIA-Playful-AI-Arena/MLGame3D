"""
MLPlay Module

This module provides a base class for implementing MLPlay classes.
"""

import numpy as np
from typing import Dict, Any


def get_default_action(action_space_info):
    """
    Get the default (no-op) action for an action space: all zeros.

    This is the action used whenever an MLPlay instance cannot provide one
    (e.g. it failed to initialize, raised an exception in update(), timed out
    or returned None).

    Args:
        action_space_info: Information about the action space

    Returns:
        A zero action matching the action space (continuous array, discrete
        int32 array, or a (continuous, discrete) tuple for hybrid spaces)
    """
    if action_space_info.is_continuous():
        return np.zeros(action_space_info.continuous_size)
    elif action_space_info.is_discrete():
        return np.zeros(action_space_info.discrete_size, dtype=np.int32)
    else:
        # Hybrid action space
        return (
            np.zeros(action_space_info.continuous_size),
            np.zeros(action_space_info.discrete_size, dtype=np.int32)
        )


class DefaultActionMLPlay:
    """
    A class that always takes the default (no-op) action.

    Used as a stand-in when a user's MLPlay class fails to initialize, so the
    player behaves the same way as when update() fails.
    """

    def __init__(self, action_space_info, name: str = "DefaultActionMLPlay", parameters: Dict[str, Any] = None):
        """
        Initialize the default-action MLPlay instance.

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
        Ignore observations and return the default action.

        Args:
            observations: A dictionary of observations
            done: Whether the episode is done
            info: Additional information

        Returns:
            The default (all-zero) action
        """
        return get_default_action(self.action_space_info)


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
