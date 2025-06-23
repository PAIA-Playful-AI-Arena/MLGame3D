"""
Keyboard State Side Channel Module

This module provides a side channel for receiving keyboard state information from Unity.
"""

import uuid
import json
from typing import Set
from mlagents_envs.side_channel import SideChannel, IncomingMessage

class KeyboardStateSideChannel(SideChannel):
    """
    A side channel for receiving keyboard state information from Unity.
    
    This side channel receives the keyboard state from Unity and provides
    methods for accessing the pressed keys.
    """
    
    def __init__(self) -> None:
        """
        Initialize the keyboard state side channel.
        """
        # Use a unique channel ID
        channel_id = uuid.UUID("a7530202-5d9b-4089-91f7-eb9e5a4b9c6f")
        super().__init__(channel_id)
        self.pressed_keys: Set[str] = set()
        self.has_changed = False
    
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
            data = json.loads(json_str)
            
            # Check if this is a keyboard state message
            if "keyboard_state" in data:
                # Get the new pressed keys
                new_pressed_keys = set(data["keyboard_state"])
                
                # Check if the state has changed
                if new_pressed_keys != self.pressed_keys:
                    self.has_changed = True
                    self.pressed_keys = new_pressed_keys
                else:
                    self.has_changed = False
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_str}")
    
    def get_pressed_keys(self) -> Set[str]:
        """
        Get the set of currently pressed keys.
        
        Returns:
            A set of strings representing the pressed keys.
        """
        return self.pressed_keys
    
    def has_keyboard_state_changed(self) -> bool:
        """
        Check if the keyboard state has changed since the last message.
        
        Returns:
            True if the keyboard state has changed, False otherwise.
        """
        return self.has_changed
    
    def reset_change_flag(self) -> None:
        """
        Reset the change flag.
        """
        self.has_changed = False