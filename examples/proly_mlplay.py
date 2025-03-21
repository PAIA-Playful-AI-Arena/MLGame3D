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
        
        # Item-related attributes
        self.inventory_items = []
        self.selected_item_index = -1
        self.nearby_items = []
        self.inventory_item_count = 0
        
        # Item usage cooldowns
        self.item_use_cooldown = 0
        self.item_selection_cooldown = 0
        
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
        
        # Reset item-related attributes
        self.inventory_items = []
        self.selected_item_index = -1
        self.nearby_items = []
        self.inventory_item_count = 0
        
        # Reset cooldowns
        self.item_use_cooldown = 0
        self.item_selection_cooldown = 0
        
    def update(self, 
               observations: Dict[str, np.ndarray], 
               reward: float = 0.0, 
               done: bool = False, 
               info: Dict[str, Any] = None) -> np.ndarray:
        """
        Process observations and choose an action.
        
        This method implements a strategy to navigate towards the target (next checkpoint)
        while avoiding obstacles, collecting items, and using items strategically.
        
        Args:
            observations: A dictionary of observations from the Proly game
            reward: The reward received
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            The action to take as a tuple of (continuous, discrete) actions
        """
        self.step_counter += 1
        
        # Parse observations
        self._parse_observations(observations)
        
        # Calculate direction to target
        direction_to_target = self._calculate_direction_to_target()
        
        # Avoid other players
        avoidance_vector = self._calculate_avoidance_vector()
        
        # Calculate pickup vector for nearby items
        pickup_vector, pickup_priority = self._calculate_pickup_vector()
        
        # Combine all vectors with appropriate weights
        # Base weight is towards the target
        combined_vector = direction_to_target
        
        # Add avoidance vector with high priority (safety first)
        if np.linalg.norm(avoidance_vector) > 0:
            combined_vector += 1.5 * avoidance_vector
        
        # Add pickup vector if there's a valuable item nearby
        if pickup_priority > 0:
            # Adjust weight based on item priority
            pickup_weight = min(1.5, pickup_priority)
            combined_vector += pickup_weight * pickup_vector
        
        # Normalize the combined vector
        if np.linalg.norm(combined_vector) > 0:
            combined_vector = combined_vector / np.linalg.norm(combined_vector)
        
        # Process items and decide on discrete actions
        select_item_action, use_item_action = self._process_items()
        
        # Decrease cooldowns
        if self.item_use_cooldown > 0:
            self.item_use_cooldown -= 1
            
        if self.item_selection_cooldown > 0:
            self.item_selection_cooldown -= 1
        
        # Unity ML-Agents 支持混合動作空間，我們需要返回一個 tuple
        # 包含 (連續動作, 離散動作)
        
        # 連續動作: 移動方向向量 [move_x, move_z]
        continuous_actions = np.array([combined_vector[0], combined_vector[1]])
        
        # 離散動作: 選擇物品和使用物品 [select_item, use_item]
        discrete_actions = np.array([select_item_action, use_item_action], dtype=np.int32)
        
        # 返回一個 tuple (連續動作, 離散動作)
        return (continuous_actions, discrete_actions)
    
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
            
        # Parse inventory items
        if "inventory_item_count" in observations:
            self.inventory_item_count = int(observations["inventory_item_count"])
            
        if "inventory_items" in observations:
            self.inventory_items = observations["inventory_items"]
            
        if "selected_item_index" in observations:
            self.selected_item_index = int(observations["selected_item_index"])
            
        if "nearby_items" in observations:
            self.nearby_items = observations["nearby_items"]
    
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
    
    def _get_item_name(self, item_type: int, item_id: int) -> str:
        """
        Get a human-readable name for an item based on its type and ID.
        
        Item type to item name mapping:
        - Type 1 (Usable): 
            ID 1: HealCake
            ID 2: SpeedCoffee
            ID 3: Shield
        - Type 2 (Throwable): 
            ID 1: Bomb
            ID 2: Rock
            ID 3: SpiderWeb
        - Type 3 (Equipment): 
            ID 1: Stick
        
        Args:
            item_type: The type of the item (1: Usable, 2: Throwable, 3: Equipment)
            item_id: The ID of the item within its type
            
        Returns:
            A string name for the item
        """
        item_type_names = ["Unknown", "Usable", "Throwable", "Equipment"]
        type_name = item_type_names[item_type] if 0 <= item_type < len(item_type_names) else "Unknown"
        
        if item_type == 1:  # Usable
            if item_id == 1:
                return "HealCake"
            elif item_id == 2:
                return "SpeedCoffee"
            elif item_id == 3:
                return "Shield"
        elif item_type == 2:  # Throwable
            if item_id == 1:
                return "Bomb"
            elif item_id == 2:
                return "Rock"
            elif item_id == 3:
                return "SpiderWeb"
        elif item_type == 3:  # Equipment
            if item_id == 1:
                return "Stick"
                
        return f"{type_name}({item_id})"
    
    def _process_items(self) -> tuple[int, int]:
        """
        Process item-related information and decide whether to select or use items.
        
        Returns:
            A tuple of (select_item_action, use_item_action) where:
              - select_item_action: 1 if we should select a different item, 0 otherwise
              - use_item_action: 1 if we should use the currently selected item, 0 otherwise
        """
        select_item_action = 0
        use_item_action = 0
        
        # Strategies for different health levels
        health_percentage = self.current_health / max(1.0, self.max_health)
        
        # Check if we have items in inventory
        if self.inventory_item_count > 0:
            # Identify HealCake (Usable item with ID 1) when health is low
            if health_percentage < 0.7 and self.item_use_cooldown <= 0:
                # Search for HealCake in inventory
                for i, item in enumerate(self.inventory_items):
                    # Check if item is Usable (type 1) and is Shield (ID 3)
                    if (item["item_type"] == 1 and item["item_id"] == 3):
                        # If we're not on this item, select it
                        if self.selected_item_index != i and self.item_selection_cooldown <= 0:
                            select_item_action = 1
                            self.item_selection_cooldown = 5  # Cooldown to avoid rapid switching
                        # If we're already on this item, use it
                        elif self.selected_item_index == i:
                            use_item_action = 1
                            self.item_use_cooldown = 10  # Cooldown to avoid spamming
                        return select_item_action, use_item_action

            # Use Throwable items like Bomb (ID 1) or Rock (ID 2) against nearby players
            if self.other_players_info and self.item_use_cooldown <= 0:
                # Find the closest player
                closest_player = min(self.other_players_info, 
                                   key=lambda p: np.linalg.norm(p["relative_position"]))
                closest_distance = np.linalg.norm(closest_player["relative_position"])
                
                # If there's a player nearby
                if closest_distance < 5.0:
                    # Search for throwable items
                    for i, item in enumerate(self.inventory_items):
                        # Check if item is Throwable (type 2)
                        if item["item_type"] == 2:
                            # If we're not on this item, select it
                            if self.selected_item_index != i and self.item_selection_cooldown <= 0:
                                select_item_action = 1
                                self.item_selection_cooldown = 5
                            # If we're already on this item, use it
                            elif self.selected_item_index == i:
                                use_item_action = 1
                                self.item_use_cooldown = 20
                            return select_item_action, use_item_action
                        
            # Use SpeedCoffee (Usable item with ID 2) when approaching checkpoints
            if health_percentage > 0.7 and self.item_use_cooldown <= 0:
                distance_to_target = np.linalg.norm(self.target_position[:2] - self.current_position[:2])
                
                # If we're far from the target but heading towards it
                if 5.0 < distance_to_target < 30.0:
                    # Search for SpeedCoffee in inventory
                    for i, item in enumerate(self.inventory_items):
                        # Check if item is Usable (type 1) and is SpeedCoffee (ID 2)
                        if (item["item_type"] == 1 and item["item_id"] == 2):
                            # If we're not on this item, select it
                            if self.selected_item_index != i and self.item_selection_cooldown <= 0:
                                select_item_action = 1
                                self.item_selection_cooldown = 5
                            # If we're already on this item, use it
                            elif self.selected_item_index == i:
                                use_item_action = 1
                                self.item_use_cooldown = 30  # Longer cooldown for speed boost
                            return select_item_action, use_item_action
        
        # Return the selected actions
        return select_item_action, use_item_action
        
    def _calculate_pickup_vector(self) -> tuple[np.ndarray, float]:
        """
        Calculate a vector towards the most valuable nearby item to pick up.
        
        Returns:
            A tuple containing:
                - A 2D direction vector (x, z) towards the item
                - A priority value (0.0-2.0) indicating how valuable the item is
        """
        # Default: no item to pick up
        pickup_vector = np.zeros(2)
        pickup_priority = 0.0
        
        # Only process if we have space in inventory and there are nearby items
        if not self.nearby_items or self.inventory_item_count >= 3:
            return pickup_vector, pickup_priority
        
        # Track the best item to pick up
        best_item = None
        best_priority = 0.0
        best_distance = float('inf')
        
        # Check each nearby item
        for nearby_item in self.nearby_items:
            if not isinstance(nearby_item, dict) or "relative_position" not in nearby_item:
                continue
                
            # Skip if missing key information
            if "item_type" not in nearby_item or "item_id" not in nearby_item:
                continue
                
            # Calculate base priority based on item type and ID
            item_priority = self._calculate_item_priority(nearby_item)
            
            # Calculate distance to item
            relative_pos = nearby_item["relative_position"]
            distance = np.linalg.norm(relative_pos)
            
            # Only consider items that are reasonably close
            if distance > 15.0:
                continue
                
            # Increase priority for closer items
            distance_factor = 1.0 - (distance / 15.0)  # 0.0-1.0 based on distance
            adjusted_priority = item_priority * distance_factor
            
            # If this item is better than our current best, update
            if adjusted_priority > best_priority or (adjusted_priority == best_priority and distance < best_distance):
                best_item = nearby_item
                best_priority = adjusted_priority
                best_distance = distance
        
        # If we found a good item, calculate direction to it
        if best_item is not None:
            relative_pos = best_item["relative_position"]
            if np.linalg.norm(relative_pos) > 0:
                pickup_vector = relative_pos / np.linalg.norm(relative_pos)
                
            # Set the final priority based on item value and distance
            pickup_priority = best_priority
            
            # Get item information for logging
            item_type = int(best_item["item_type"])
            item_id = int(best_item["item_id"])
            item_name = self._get_item_name(item_type, item_id)
            
            # Debug message
            if self.step_counter % 20 == 0:  # Only print every 20 steps to avoid spamming
                print(f"Moving toward {item_name} at distance {best_distance:.1f} with priority {pickup_priority:.1f}")
        
        return pickup_vector, pickup_priority
    
    def _calculate_item_priority(self, item: Dict[str, Any]) -> float:
        """
        Calculate the priority of an item based on its type and ID.
        
        Item type to item name mapping:
        - Type 1 (Usable): 
            ID 1: HealCake
            ID 2: SpeedCoffee
            ID 3: Shield
        - Type 2 (Throwable): 
            ID 1: Bomb
            ID 2: Rock
            ID 3: SpiderWeb
        - Type 3 (Equipment): 
            ID 1: Stick
        
        Args:
            item: The item dictionary from observations
            
        Returns:
            A priority value between 0.0 and 2.0
        """
        item_type = int(item["item_type"])
        item_id = int(item["item_id"])
        health_percentage = self.current_health / max(1.0, self.max_health)
        
        # Base priority
        priority = 1.0
        
        # Check for item already in inventory
        have_similar_item = False
        for inv_item in self.inventory_items:
            if inv_item["item_type"] == item_type and inv_item["item_id"] == item_id:
                have_similar_item = True
                break
        
        # Get item name for better logging
        item_name = self._get_item_name(item_type, item_id)
        
        # Adjustment based on item type
        if item_type == 1:  # Usable items
            if item_id == 1:  # HealCake
                # Higher priority when health is low
                if health_percentage < 0.3:
                    priority = 2.0
                elif health_percentage < 0.7:
                    priority = 1.5
                else:
                    priority = 1.0
                    
            elif item_id == 2:  # SpeedCoffee
                # Higher priority when health is good (we'd use it for speed)
                if health_percentage > 0.7:
                    priority = 1.5
                else:
                    priority = 0.8
                    
            elif item_id == 3:  # Shield
                # Always useful for protection
                priority = 1.2
        
        elif item_type == 2:  # Throwable items
            if self.other_players_info:  # More valuable when other players are around
                if item_id == 1:  # Bomb
                    priority = 1.4
                elif item_id == 2:  # Rock
                    priority = 1.1
                elif item_id == 3:  # SpiderWeb
                    priority = 1.2
            else:
                # Less valuable when playing alone
                priority = 0.7
                
        elif item_type == 3:  # Equipment items
            if item_id == 1:  # Stick
                priority = 1.3
        
        # Reduce priority if we already have this item
        if have_similar_item:
            priority *= 0.5
            
        # Adjust based on inventory fullness
        if self.inventory_item_count == 2:  # Almost full inventory
            # Be more selective
            priority *= 0.8
        
        return priority
