"""
Agent Module

This module provides a base class for implementing game agents that can interact with Unity games.
"""

import numpy as np
from typing import Dict, Any, Callable
from mlagents_envs.base_env import ActionSpec

class Agent:
    """
    Base class for implementing game agents that can interact with Unity games.
    """
    
    def __init__(self, name: str = "Agent"):
        """
        Initialize the agent.
        
        Args:
            name: The name of the agent
        """
        self.name = name
        self.total_reward = 0.0
        self.episode_count = 0
        self.step_count = 0
    
    def reset(self) -> None:
        """
        Reset the agent for a new episode.
        """
        self.step_count = 0
    
    def act(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose an action based on the current observations.
        
        Args:
            observations: A dictionary of observations
            
        Returns:
            The action to take
        """
        raise NotImplementedError("Subclasses must implement act()")
    
    def observe(self, observations: Dict[str, np.ndarray], reward: float, done: bool, info: Dict[str, Any]) -> None:
        """
        Process the observations, reward, and other information from the environment.
        
        Args:
            observations: A dictionary of observations
            reward: The reward received
            done: Whether the episode is done
            info: Additional information
        """
        self.total_reward += reward
        self.step_count += 1
        
        if done:
            self.episode_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's performance.
        
        Returns:
            A dictionary of statistics
        """
        return {
            "name": self.name,
            "total_reward": self.total_reward,
            "episode_count": self.episode_count,
            "average_reward": self.total_reward / max(1, self.episode_count),
            "step_count": self.step_count
        }


class RandomAgent(Agent):
    """
    An agent that takes random actions.
    """
    
    def __init__(self, action_space_info: ActionSpec, name: str = "RandomAgent"):
        """
        Initialize the random agent.
        
        Args:
            action_space_info: Information about the action space
            name: The name of the agent
        """
        super().__init__(name)
        self.action_space_info = action_space_info
    
    def act(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose a random action.
        
        Args:
            observations: A dictionary of observations
            
        Returns:
            A random action
        """
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


class KeyboardAgent(Agent):
    """
    An agent that takes actions based on keyboard input.
    This is a base class that should be subclassed for specific games.
    """
    
    def __init__(self, action_space_info: ActionSpec, name: str = "KeyboardAgent"):
        """
        Initialize the keyboard agent.
        
        Args:
            action_space_info: Information about the action space
            name: The name of the agent
        """
        super().__init__(name)
        self.action_space_info = action_space_info
        
        # This should be overridden in subclasses
        self.key_action_map = {'w': np.array([0, 1]), 's': np.array([0, -1]), 'a': np.array([-1, 0]), 'd': np.array([1, 0])}
    
    def act(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose an action based on keyboard input.
        
        Args:
            observations: A dictionary of observations
            
        Returns:
            The action corresponding to the keyboard input
        """
        # This is a placeholder. Subclasses should implement their own keyboard input handling.
        # For example, using a library like pygame or pynput to get keyboard input.
        key = input("Enter action key: ")
        
        if key in self.key_action_map:
            return self.key_action_map[key]
        else:
            # Return a default action if the key is not mapped
            if self.action_space_info.is_continuous():
                return np.zeros(self.action_space_info.continuous_size)
            elif self.action_space_info.is_discrete():
                return np.zeros(len(self.action_space_info.discrete_branches), dtype=np.int32)
            else:
                continuous = np.zeros(self.action_space_info.continuous_size)
                discrete = np.zeros(len(self.action_space_info.discrete_branches), dtype=np.int32)
                return (continuous, discrete)


class HumanAgent(Agent):
    """
    An agent that delegates action selection to a human through a GUI.
    This is a base class that should be subclassed for specific games.
    """
    
    def __init__(
        self, 
        action_space_info: ActionSpec, 
        get_action_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        name: str = "HumanAgent"
    ):
        """
        Initialize the human agent.
        
        Args:
            action_space_info: Information about the action space
            get_action_fn: A function that gets an action from the human through a GUI
            name: The name of the agent
        """
        super().__init__(name)
        self.action_space_info = action_space_info
        self.get_action_fn = get_action_fn
    
    def act(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get an action from the human through the GUI.
        
        Args:
            observations: A dictionary of observations
            
        Returns:
            The action chosen by the human
        """
