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
        self.target_position = observations["target_position"]
        self.current_position = observations["agent_position"]
        self.current_velocity = observations["agent_velocity"]
        
        if "agent_health" in observations:
            self.current_health = observations["agent_health"]
            
        if "agent_health_normalized" in observations:
            self.max_health = self.current_health / max(0.01, observations["agent_health_normalized"])
            
        if "last_checkpoint_index" in observations:
            self.last_checkpoint_index = int(observations["last_checkpoint_index"])
            
        if "current_time" in observations:
            self.current_time = observations["current_time"]
            
        if "other_players" in observations:
            self.other_players_info = observations["other_players"]
    
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
