"""
Game Environment Module

This module provides a wrapper around the Unity ML-Agents environment
to simplify the interaction with Unity games.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, ActionSpec
from mlgame3d.side_channel.observation_structure_side_channel import ObservationStructureSideChannel
from mlgame3d.side_channel.player_control_side_channel import PlayerControlSideChannel
from mlgame3d.side_channel.game_parameters_side_channel import GameParametersSideChannel

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
        timeout_wait: int = 60,
        controlled_players: List[int] = [],
        control_modes: List[str] = [],
        game_parameters: Optional[List[Tuple[str, Any]]] = None
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
            controlled_players: List of player IDs to control.
            control_modes: List of control modes ("manual" or "mlplay") for each player.
        """
        # Create the observation structure side channel
        self.observation_structure_side_channel = ObservationStructureSideChannel()

        # Initialize the player control side channel
        self.player_control_channel = PlayerControlSideChannel()
        
        # Initialize the game parameters side channel
        self.game_parameters_channel = GameParametersSideChannel()
        
        # Initialize the Unity environment with the side channel
        self.env = UnityEnvironment(
            file_name=file_name,
            worker_id=worker_id,
            base_port=base_port,
            seed=seed,
            no_graphics=no_graphics,
            timeout_wait=timeout_wait,
            side_channels=[
                self.observation_structure_side_channel,
                self.player_control_channel,
                self.game_parameters_channel
            ]
        )

        # Set all players to be controlled with their respective modes
        self.player_control_channel.set_controlled_players(controlled_players, control_modes)

        # Process game parameters if provided
        if game_parameters is not None:
            game_params = {}
            for key, value in game_parameters:
                # Try to convert value to appropriate type
                try:
                    # Try as int
                    game_params[key] = int(value)
                except ValueError:
                    try:
                        # Try as float
                        game_params[key] = float(value)
                    except ValueError:
                        # Try as boolean
                        if value.lower() in ('true', 'yes', '1'):
                            game_params[key] = True
                        elif value.lower() in ('false', 'no', '0'):
                            game_params[key] = False
                        else:
                            # Keep as string
                            game_params[key] = value
                        
            if game_params:
                self.set_game_parameters(game_params)

        # Initialize environment
        self.env.reset()
        
        # Request the observation structure from Unity
        if not self.observation_structure_side_channel.has_observation_structure():
            self.observation_structure_side_channel.request_observation_structure()

        self.behavior_names = sorted(list(self.env.behavior_specs.keys()))
        if not self.behavior_names:
            raise ValueError("No behaviors found in the environment")
        
        # Store behavior specs for easy access
        self.behavior_specs = self.env.behavior_specs
        
        # Get initial observations
        for behavior_name in self.behavior_names:
            self.decision_steps, self.terminal_steps = self.env.get_steps(behavior_name)
        
    def reset(self) -> List[Dict[str, np.ndarray]]:
        """
        Reset the environment and return the initial observations.
        
        Returns:
            A list of dictionaries of initial observations for each agent.
        """
        self.env.reset()
        observations = {}
        for behavior_name in self.behavior_names:
            self.decision_steps, self.terminal_steps = self.env.get_steps(behavior_name)

            if (len(self.decision_steps) == 0):
                # If there are no agents, return empty observations
                observations[behavior_name] = {}
                continue
        
            # Return observations for each agent
            agent_id = self.decision_steps.agent_id[0]
            observations[behavior_name] = self._get_obs_dict_for_agent(self.decision_steps, agent_id)
                
        return observations
    
    def step(self, actions: Dict[str, List[np.ndarray]]) -> Tuple[List[Dict[str, np.ndarray]], List[float], bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            actions: A list of actions to take for each agent.
            
        Returns:
            A tuple containing:
                - observations: A list of dictionaries of observations for each agent
                - rewards: A list of rewards received by each agent
                - done: Whether the episode is done
                - info: Additional information
        """
        if actions:
            for behavior_name in self.behavior_names:
                if behavior_name not in actions or len(actions[behavior_name]) == 0:
                    continue
                    
                # Create a combined action tuple for all agents
                action_tuple = self._create_combined_action_tuple(actions, behavior_name)
                
                # Set the actions for the default behavior
                self.env.set_actions(behavior_name, action_tuple)
            
        # Step the environment
        self.env.step()

        observations = {}
        rewards = {}
        info = {}
        
        for behavior_name in self.behavior_names:
            # Get the new decision steps and terminal steps
            self.decision_steps, self.terminal_steps = self.env.get_steps(behavior_name)
            
            # Check if the episode is done
            done = len(self.terminal_steps) > 0
            
            if done:
                # If the episode is done, get observations from terminal steps
                agent_id = self.terminal_steps.agent_id[0]
                observations[behavior_name] = self._get_obs_dict_for_agent(self.terminal_steps, agent_id)
                rewards[behavior_name] = self.terminal_steps.reward[0]
                info = {"interrupted": [self.terminal_steps.interrupted[0]]}
            else:
                # If the episode is not done, get observations from decision steps
                if (len(self.decision_steps) == 0):
                    # If there are no agents, return empty observations
                    observations[behavior_name] = {}
                    rewards[behavior_name] = 0.0
                else:
                    agent_id = self.decision_steps.agent_id[0]
                    observations[behavior_name] = self._get_obs_dict_for_agent(self.decision_steps, agent_id)
                    rewards[behavior_name] = self.decision_steps.reward[0]
        
        return observations, rewards, done, info
    
    def _get_obs_dict_for_agent(self, steps, agent_id) -> Dict[str, np.ndarray]:
        """
        Convert the observations from the steps to a dictionary for a specific agent.
        
        Args:
            steps: Either DecisionSteps or TerminalSteps
            agent_id: The ID of the agent
            
        Returns:
            A dictionary of observations for the agent
        """
        if agent_id not in steps:
            return {}
            
        agent_step = steps[agent_id]
        obs_dict = {}
        
        # Parse the observations using the observation structure if available
        if self.observation_structure_side_channel.has_observation_structure() and len(agent_step.obs) > 0:
            # Get the first observation (which should be the vector observation)
            vector_obs = agent_step.obs[0]
            
            # Parse the observation using the observation structure
            parsed_obs = self.observation_structure_side_channel.parse_observation(vector_obs)
            
            # Add the parsed observations to the dictionary
            for key, value in parsed_obs.items():
                obs_dict[key] = value
            
        return obs_dict
    
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
    
    def _create_combined_action_tuple(self, actions: Dict[str, List[np.ndarray]], behavior_name: str) -> ActionTuple:
        """
        Create a combined ActionTuple from the given actions for all agents.
        
        Args:
            actions: A list of actions to convert
            
        Returns:
            An ActionTuple containing the actions for all agents
        """
        action_spec = self.behavior_specs[behavior_name].action_spec
        
        if action_spec.is_continuous():
            # Continuous action space
            continuous_actions = np.vstack([action.reshape(1, -1) for action in actions[behavior_name]])
            return ActionTuple(continuous=continuous_actions)
        elif action_spec.is_discrete():
            # Discrete action space
            discrete_actions = np.vstack([action.reshape(1, -1).astype(np.int32) for action in actions[behavior_name]])
            return ActionTuple(discrete=discrete_actions)
        else:
            # Hybrid action space
            continuous_actions = []
            discrete_actions = []
            
            for action in actions[behavior_name]:
                if isinstance(action, tuple) and len(action) == 2:
                    continuous, discrete = action
                    continuous_actions.append(continuous.reshape(1, -1))
                    discrete_actions.append(discrete.reshape(1, -1).astype(np.int32))
                else:
                    raise ValueError(
                        "For hybrid action spaces, action must be a tuple of (continuous_actions, discrete_actions)"
                    )
                    
            return ActionTuple(
                continuous=np.vstack(continuous_actions),
                discrete=np.vstack(discrete_actions)
            )
    
    def close(self):
        """
        Close the environment.
        """
        self.env.close()
    
    def get_action_space_info(self, behavior_name) -> ActionSpec:
        """
        Get information about the action space.
        
        Returns:
            The action specification for the default behavior
        """
        return self.behavior_specs[behavior_name].action_spec
        
    def set_game_parameter(self, key: str, value: Any) -> None:
        """
        Set a game parameter.
        
        Args:
            key: The parameter key
            value: The parameter value
        """
        self.game_parameters_channel.set_parameter(key, value)
        
    def set_game_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set multiple game parameters at once.
        
        Args:
            parameters: Dictionary of parameter key-value pairs
        """
        self.game_parameters_channel.set_parameters(parameters)
