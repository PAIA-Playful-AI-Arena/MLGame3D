"""
Game Environment Module

This module provides a wrapper around the Unity ML-Agents environment
to simplify the interaction with Unity games.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, ActionSpec

class GameEnvironment:
    """
    A wrapper around the Unity ML-Agents environment to simplify the interaction with Unity games.
    """
    
    def __init__(
        self, 
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60
    ):
        """
        Initialize the game environment.
        
        Args:
            file_name: Path to the Unity executable. If None, will connect to an already running Unity editor.
            worker_id: Offset from base_port. Used for training multiple environments simultaneously.
            base_port: Base port to connect to Unity environment. If None, defaults to 5004 for editor or 5005 for executable.
            seed: Random seed for the environment.
            no_graphics: Whether to run the Unity simulator in no-graphics mode.
            timeout_wait: Time (in seconds) to wait for connection from environment.
        """
        self.env = UnityEnvironment(
            file_name=file_name,
            worker_id=worker_id,
            base_port=base_port,
            seed=seed,
            no_graphics=no_graphics,
            timeout_wait=timeout_wait
        )

        # Initialize environment
        self.env.reset()

        self.behavior_names = list(self.env.behavior_specs.keys())
        if not self.behavior_names:
            raise ValueError("No behaviors found in the environment")
        
        # Default to the first behavior if there are multiple
        self.default_behavior = self.behavior_names[0]
        
        # Store behavior specs for easy access
        self.behavior_specs = self.env.behavior_specs
        
        # Get initial observations
        self.decision_steps, self.terminal_steps = self.env.get_steps(self.default_behavior)
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment and return the initial observations.
        
        Returns:
            A dictionary of initial observations for the default behavior.
        """
        self.env.reset()
        self.decision_steps, self.terminal_steps = self.env.get_steps(self.default_behavior)
        return self._get_obs_dict(self.decision_steps)
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take.
            
        Returns:
            A tuple containing:
                - observations: A dictionary of observations
                - reward: The reward received
                - done: Whether the episode is done
                - info: Additional information
        """
        # Convert the action to an ActionTuple
        action_tuple = self._create_action_tuple(action)
        
        # Set the actions for the default behavior
        self.env.set_actions(self.default_behavior, action_tuple)
        
        # Step the environment
        self.env.step()
        
        # Get the new decision steps and terminal steps
        self.decision_steps, self.terminal_steps = self.env.get_steps(self.default_behavior)
        
        # Check if the episode is done
        done = len(self.terminal_steps) > 0
        
        # Get the observations, reward, and info
        if done:
            obs = self._get_obs_dict(self.terminal_steps)
            reward = self.terminal_steps.reward[0] if len(self.terminal_steps) > 0 else 0.0
            info = {"interrupted": self.terminal_steps.interrupted[0] if len(self.terminal_steps) > 0 else False}
        else:
            obs = self._get_obs_dict(self.decision_steps)
            reward = self.decision_steps.reward[0] if len(self.decision_steps) > 0 else 0.0
            info = {}
        
        return obs, reward, done, info
    
    def _get_obs_dict(self, steps) -> Dict[str, np.ndarray]:
        """
        Convert the observations from the steps to a dictionary.
        
        Args:
            steps: Either DecisionSteps or TerminalSteps
            
        Returns:
            A dictionary of observations
        """
        if len(steps) == 0:
            # If there are no agents, return empty observations
            return {}
        
        obs_dict = {}
        for i, obs in enumerate(steps.obs):
            obs_dict[f"obs_{i}"] = obs[0]  # Take the first agent's observation
        
        return obs_dict
    
    def _create_action_tuple(self, action: np.ndarray) -> ActionTuple:
        """
        Create an ActionTuple from the given action.
        
        Args:
            action: The action to convert
            
        Returns:
            An ActionTuple containing the action
        """
        action_spec = self.behavior_specs[self.default_behavior].action_spec
        
        if action_spec.is_continuous():
            # Continuous action space
            continuous_actions = action.reshape(1, -1)
            return ActionTuple(continuous=continuous_actions)
        elif action_spec.is_discrete():
            # Discrete action space
            discrete_actions = action.reshape(1, -1).astype(np.int32)
            return ActionTuple(discrete=discrete_actions)
        else:
            # Hybrid action space
            if isinstance(action, tuple) and len(action) == 2:
                continuous_actions, discrete_actions = action
                return ActionTuple(
                    continuous=continuous_actions.reshape(1, -1),
                    discrete=discrete_actions.reshape(1, -1).astype(np.int32)
                )
            else:
                raise ValueError(
                    "For hybrid action spaces, action must be a tuple of (continuous_actions, discrete_actions)"
                )
    
    def close(self):
        """
        Close the environment.
        """
        self.env.close()
    
    def get_action_space_info(self) -> ActionSpec:
        """
        Get information about the action space.
        
        Returns:
            A dictionary containing information about the action space
        """
        return self.behavior_specs[self.default_behavior].action_spec
        
