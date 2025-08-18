"""
Observation Structure Side Channel Module

This module provides a side channel for receiving the observation structure from Unity.
"""

import uuid
import json
import numpy as np
from typing import Dict, Any
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage

class ObservationStructureSideChannel(SideChannel):
    """
    A side channel for receiving the observation structure from Unity.
    
    This side channel receives the observation structure from Unity and provides
    methods for parsing observations based on that structure.
    """
    
    def __init__(self) -> None:
        """
        Initialize the observation structure side channel.
        """
        # Use the same channel ID as in the Unity side
        channel_id = uuid.UUID("63e48524-4c56-40c4-bcb7-d05bc8346f4d")
        super().__init__(channel_id)
        self.observation_structure = None
        self.has_requested_structure = False
    
    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Called when a message is received from Unity.
        
        Args:
            msg: The incoming message from Unity.
        """
        # Read the message as a string
        json_str = msg.read_string()
        
        try:
            # Parse the JSON string
            self.observation_structure = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_str}")
    
    def request_observation_structure(self) -> None:
        """
        Request the observation structure from Unity.
        """
        if not self.has_requested_structure:
            # Create an outgoing message
            outgoing_msg = OutgoingMessage()
            outgoing_msg.write_string("REQUEST_OBSERVATION_STRUCTURE")
            self.queue_message_to_send(outgoing_msg)
            self.has_requested_structure = True
            print("Requested observation structure from Unity")
    
    def has_observation_structure(self) -> bool:
        """
        Check if the observation structure has been received.
        
        Returns:
            True if the observation structure has been received, False otherwise.
        """
        return self.observation_structure is not None
    
    def parse_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Parse an observation based on the received observation structure.
        
        Args:
            observation: The observation to parse.
            
        Returns:
            A dictionary containing the parsed observation.
        """
        if not self.has_observation_structure():
            print("Warning: Observation structure not received yet. Cannot parse observation.")
            return {"obs_1": observation}
        
        # Parse the observation recursively
        return self._parse_observation_recursive(self.observation_structure, observation, 0)[0]
    
    def _parse_observation_recursive(self, structure, observation: np.ndarray, start_index: int) -> tuple[Dict[str, Any], int]:
        """
        Recursively parse an observation based on the structure.
        
        Args:
            structure: The structure to use for parsing.
            observation: The observation to parse.
            start_index: The index to start parsing from.
            
        Returns:
            A tuple containing:
                - A dictionary containing the parsed observation.
                - The next index to parse from.
        """
        parsed_obs = {}
        current_index = start_index
        
        # If structure is a list, process each item
        if isinstance(structure, list):
            for item in structure:
                key = item.get("key", "")
                if not key:
                    continue
                
                # Parse the item based on its type
                item_result, current_index = self._parse_item(item, observation, current_index)
                parsed_obs[key] = item_result
        # If structure is a dictionary, process it as a single item
        elif isinstance(structure, dict):
            key = structure.get("key", "")
            if key:
                item_result, current_index = self._parse_item(structure, observation, current_index)
                parsed_obs[key] = item_result
        
        return parsed_obs, current_index
    
    def _calculate_item_size(self, items) -> int:
        """
        Calculate the size of a list item recursively.
        
        Args:
            items: The list of items to calculate the size for.
            
        Returns:
            The size of the list item.
        """
        total_size = 0
        
        for item in items:
            item_type = item.get("type", "")
            
            if item_type == "Vector3":
                total_size += 3
            elif item_type == "Vector2":
                total_size += 2
            elif item_type == "float" or item_type == "int" or item_type == "bool":
                total_size += 1
            elif item_type == "Vector":
                total_size += item.get("vector_size", 0)
            elif item_type == "Grid":
                # For grid, calculate based on grid_size and items
                grid_size = item.get("grid_size", 0)
                sub_items = item.get("items", [])
                sub_item_size = self._calculate_item_size(sub_items)
                
                # Grid is a 2D structure, so we square the grid_size
                total_size += sub_item_size * grid_size * grid_size
            elif item_type == "List":
                # For nested lists, calculate recursively
                sub_items = item.get("items", [])
                sub_item_size = self._calculate_item_size(sub_items)
                sub_item_count = item.get("item_count", 0)
                
                # If item_count is specified, use it, otherwise assume 1
                if sub_item_count > 0:
                    total_size += sub_item_size * sub_item_count
                else:
                    total_size += sub_item_size
        
        return total_size
    
    def _parse_item(self, item: Dict[str, Any], observation: np.ndarray, start_index: int) -> tuple[Any, int]:
        """
        Parse a single item from the observation.
        
        Args:
            item: The item structure to use for parsing.
            observation: The observation to parse.
            start_index: The index to start parsing from.
            
        Returns:
            A tuple containing:
                - The parsed item.
                - The next index to parse from.
        """
        item_type = item.get("type", "")
        current_index = start_index
        
        # Handle different types
        if item_type == "Vector3":
            # Vector3 has 3 components (x, y, z)
            if current_index + 3 <= len(observation):
                result = observation[current_index:current_index + 3]
                current_index += 3
            else:
                print(f"Warning: Not enough data for {item.get('key', '')} (Vector3)")
                result = np.zeros(3)
        elif item_type == "Vector2":
            # Vector2 has 2 components (x, y)
            if current_index + 2 <= len(observation):
                result = observation[current_index:current_index + 2]
                current_index += 2
            else:
                print(f"Warning: Not enough data for {item.get('key', '')} (Vector2)")
                result = np.zeros(2)
        elif item_type == "float" or item_type == "int" or item_type == "bool":
            # float, int and bool are single values
            if current_index < len(observation):
                if item_type == "int":
                    result = int(observation[current_index])
                elif item_type == "bool":
                    result = bool(observation[current_index] > 0.5)  # Convert float to bool using threshold
                else:
                    result = observation[current_index]
                current_index += 1
            else:
                print(f"Warning: Not enough data for {item.get('key', '')} ({item_type})")
                if item_type == "bool":
                    result = False
                else:
                    result = 0
        elif item_type == "Vector":
            vector_size = item.get("vector_size", 0)
            if current_index + vector_size <= len(observation):
                result = observation[current_index:current_index + vector_size]
                current_index += vector_size
            else:
                print(f"Warning: Not enough data for {item.get('key', '')} (Vector)")
                result = np.zeros(vector_size)
        elif item_type == "List":
            # Handle list of items
            items = item.get("items", [])
            item_count = item.get("item_count", 0)
            
            # Calculate the size of each list item using recursion
            item_size = self._calculate_item_size(items)
            
            # Use the specified item_count if available, otherwise calculate from remaining data
            num_items = item_count
            if num_items == 0:
                # Calculate how many items we can parse
                remaining_obs_length = len(observation) - current_index
                num_items = remaining_obs_length // item_size if item_size > 0 else 0
            
            # Parse each item in the list
            list_items = []
            for i in range(num_items):
                item_data = {}
                for sub_item in items:
                    sub_key = sub_item.get("key", "")
                    if not sub_key:
                        continue
                    
                    # Parse the sub-item recursively
                    sub_result, next_index = self._parse_item(sub_item, observation, current_index)
                    item_data[sub_key] = sub_result
                    current_index = next_index
                
                list_items.append(item_data)
            
            result = list_items
        elif item_type == "Grid":
            # Handle grid of items
            grid_size = item.get("grid_size", 0)
            items = item.get("items", [])
            
            # Create a 2D grid structure
            grid_data = np.zeros((grid_size, grid_size), dtype=object)
            
            # Parse each cell in the grid
            for row in range(grid_size):
                for col in range(grid_size):
                    cell_data = {}
                    for sub_item in items:
                        sub_key = sub_item.get("key", "")
                        if not sub_key:
                            continue
                        
                        # Parse the sub-item recursively
                        sub_result, next_index = self._parse_item(sub_item, observation, current_index)
                        cell_data[sub_key] = sub_result
                        current_index = next_index
                    
                    grid_data[row, col] = cell_data
            
            result = grid_data
        else:
            # Unknown type, skip
            print(f"Warning: Unknown type {item_type} for {item.get('key', '')}")
            result = None
        
        return result, current_index
