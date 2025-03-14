"""
Custom Agent Example

This file demonstrates how to create a custom agent for the MLGame3D framework.
"""

import numpy as np
from typing import Dict, Any
from mlgame3d.agent import Agent

class CustomAgent(Agent):
    """
    A custom agent that demonstrates how to implement a simple strategy.
    """
    
    def __init__(self, action_space_info, name: str = "CustomAgent"):
        """
        Initialize the custom agent.
        
        Args:
            action_space_info: Information about the action space
            name: The name of the agent
        """
        super().__init__(name)
        self.action_space_info = action_space_info
        self.step_counter = 0
        
    def reset(self) -> None:
        """
        Reset the agent for a new episode.
        """
        super().reset()
        self.step_counter = 0
        
    def act(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose an action based on the current observations.
        
        This is a simple example that alternates between different actions
        based on the step counter.
        
        Args:
            observations: A dictionary of observations
            
        Returns:
            The action to take
        """
        self.step_counter += 1
        
        # Example of how to use observations
        # Print the first observation every 10 steps
        if self.step_counter % 10 == 0 and "obs_0" in observations:
            print(f"Step {self.step_counter}, Observation shape: {observations['obs_0'].shape}")
        
        # Different action strategies based on the action space type
        if self.action_space_info.is_continuous():
            # For continuous action spaces, create a simple pattern
            # Alternate between moving forward, backward, left, and right
            phase = (self.step_counter % 4)
            if phase == 0:
                return np.array([0.0, 1.0])  # Forward
            elif phase == 1:
                return np.array([0.0, -1.0])  # Backward
            elif phase == 2:
                return np.array([-1.0, 0.0])  # Left
            else:
                return np.array([1.0, 0.0])  # Right
                
        elif self.action_space_info.is_discrete():
            # For discrete action spaces, cycle through the possible actions
            actions = []
            for i, branch_size in enumerate(self.action_space_info.discrete_branches):
                actions.append((self.step_counter + i) % branch_size)
            return np.array(actions, dtype=np.int32)
            
        else:
            # For hybrid action spaces, combine the above strategies
            continuous_size = self.action_space_info.continuous_size
            discrete_branches = self.action_space_info.discrete_branches
            
            # Continuous part
            continuous = np.zeros(continuous_size)
            if continuous_size >= 2:
                phase = (self.step_counter % 4)
                if phase == 0:
                    continuous[0:2] = [0.0, 1.0]  # Forward
                elif phase == 1:
                    continuous[0:2] = [0.0, -1.0]  # Backward
                elif phase == 2:
                    continuous[0:2] = [-1.0, 0.0]  # Left
                else:
                    continuous[0:2] = [1.0, 0.0]  # Right
            
            # Discrete part
            discrete = np.zeros(len(discrete_branches), dtype=np.int32)
            for i, branch_size in enumerate(discrete_branches):
                discrete[i] = (self.step_counter + i) % branch_size
                
            return (continuous, discrete)
    
    def observe(self, observations: Dict[str, np.ndarray], reward: float, done: bool, info: Dict[str, Any]) -> None:
        """
        Process the observations, reward, and other information from the environment.
        
        Args:
            observations: A dictionary of observations
            reward: The reward received
            done: Whether the episode is done
            info: Additional information
        """
        super().observe(observations, reward, done, info)
        
        # Example of how to use the reward and done information
        if reward != 0:
            print(f"Received reward: {reward:.2f}")
            
        if done:
            print(f"Episode finished after {self.step_count} steps with total reward: {self.total_reward:.2f}")
