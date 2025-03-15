"""
Proly MLPlay Example

This file demonstrates how to create a custom MLPlay class for the Proly game using the MLGame3D framework.
"""

import numpy as np
from typing import Dict, Any

class MLPlay:
    """
    A custom MLPlay class designed for the Proly game.
    
    This class demonstrates how to process the observations from the Proly game
    and make decisions based on them without inheriting from any base class.
    """
    
    def __init__(self, action_space_info=None):
        """
        Initialize the MLPlay instance.
        
        Args:
            action_space_info: Information about the action space (optional)
        """
        self.step_counter = 0
        self.last_checkpoint_index = -1
        self.target_position = np.zeros(3)
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(2)
        self.current_health = 0
        self.max_health = 0
        self.current_time = 0
        self.other_players_info = []
        
    def reset(self):
        """
        Reset the agent for a new episode.
        """
        self.step_counter = 0
        self.last_checkpoint_index = -1
        self.target_position = np.zeros(3)
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(2)
        self.current_health = 0
        self.max_health = 0
        self.current_time = 0
        self.other_players_info = []
        
    def update(self, 
               observations: Dict[str, np.ndarray], 
               reward: float = 0.0, 
               done: bool = False, 
               info: Dict[str, Any] = None) -> np.ndarray:
        """
        Process observations and choose an action.
        
        This method implements a simple strategy to navigate towards the target (next checkpoint)
        while avoiding obstacles and other players.
        
        Args:
            observations: A dictionary of observations from the Proly game
            reward: The reward received
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            The action to take (continuous 2D movement vector)
        """
        self.step_counter += 1
        
        # Parse observations
        self._parse_observations(observations)
        
        # Print debug information every 100 steps
        if self.step_counter % 100 == 0:
            self._print_debug_info()
        
        # Calculate direction to target
        direction_to_target = self._calculate_direction_to_target()
        
        # Avoid other players
        avoidance_vector = self._calculate_avoidance_vector()
        
        # Combine direction and avoidance
        combined_vector = direction_to_target + avoidance_vector
        
        # Normalize the combined vector
        if np.linalg.norm(combined_vector) > 0:
            combined_vector = combined_vector / np.linalg.norm(combined_vector)
        
        return combined_vector
    
    def _parse_observations(self, observations: Dict[str, np.ndarray]) -> None:
        """
        Parse the observations from the Proly game.
        
        Args:
            observations: A dictionary of observations
        """
        if "obs_0" not in observations:
            return
            
        obs = observations["obs_0"]
        
        # The first 3 values are the target position (x, y, z)
        self.target_position = obs[0:3]
        
        # The next 3 values are the current position (x, y, z)
        self.current_position = obs[3:6]
        
        # The next 2 values are the current velocity (x, z)
        self.current_velocity = obs[6:8]
        
        # The next 2 values are the current health and normalized health
        if len(obs) > 9:
            self.current_health = obs[8]
            self.max_health = obs[8] / max(0.01, obs[9])  # Avoid division by zero
        
        # The next value is the last checkpoint index
        if len(obs) > 10:
            self.last_checkpoint_index = int(obs[10])
        
        # The next value is the current time
        if len(obs) > 11:
            self.current_time = obs[11]
        
        # The rest of the values are information about other players
        # Each player has 4 values: relative position (x, z) and relative velocity (x, z)
        if len(obs) > 12:
            self.other_players_info = []
            for i in range(12, len(obs), 4):
                if i + 4 <= len(obs):
                    self.other_players_info.append({
                        "relative_position": obs[i:i+2],
                        "relative_velocity": obs[i+2:i+4]
                    })
    
    def _calculate_direction_to_target(self) -> np.ndarray:
        """
        Calculate the direction vector to the target.
        
        Returns:
            A 2D direction vector (x, z)
        """
        # Calculate direction to target in 3D
        direction = self.target_position - self.current_position
        
        # Convert to 2D (x, z)
        direction_2d = np.array([direction[0], direction[2]])
        
        # Normalize
        if np.linalg.norm(direction_2d) > 0:
            direction_2d = direction_2d / np.linalg.norm(direction_2d)
            
        return direction_2d
    
    def _calculate_avoidance_vector(self) -> np.ndarray:
        """
        Calculate an avoidance vector to avoid other players.
        
        Returns:
            A 2D avoidance vector (x, z)
        """
        avoidance_vector = np.zeros(2)
        
        for player_info in self.other_players_info:
            relative_pos = player_info["relative_position"]
            distance = np.linalg.norm(relative_pos)
            
            # Only avoid if the player is close
            if distance < 5.0:
                # The closer the player, the stronger the avoidance
                avoidance_strength = 1.0 - (distance / 5.0)
                
                # Direction away from the player
                if distance > 0:
                    avoidance_direction = -relative_pos / distance
                    avoidance_vector += avoidance_direction * avoidance_strength
        
        # Normalize
        if np.linalg.norm(avoidance_vector) > 0:
            avoidance_vector = avoidance_vector / np.linalg.norm(avoidance_vector)
            
        return avoidance_vector
    
    def _print_debug_info(self) -> None:
        """
        Print debug information about the agent's state.
        """
        print(f"Step: {self.step_counter}")
        print(f"Position: {self.current_position}")
        print(f"Target: {self.target_position}")
        print(f"Velocity: {self.current_velocity}")
        print(f"Health: {self.current_health}/{self.max_health}")
        print(f"Last Checkpoint: {self.last_checkpoint_index}")
        print(f"Time: {self.current_time}")
        print(f"Other Players: {len(self.other_players_info)}")
        print("---")
