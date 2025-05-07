"""
Game Parameters Side Channel Module

This module provides a side channel for sending game configuration parameters to Unity.
"""

import uuid
import json
from typing import Dict, Any
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage

class GameParametersSideChannel(SideChannel):
    """
    A side channel for sending game configuration parameters to Unity.
    
    This side channel allows configuring various game parameters such as:
    - Checkpoint generation mode (random/fixed)
    - Number of checkpoints
    - Available items in the game
    - And other customizable parameters
    """
    
    def __init__(self) -> None:
        """
        Initialize the game parameters side channel.
        """
        # Use a unique channel ID
        channel_id = uuid.UUID("c2db50c5-54c5-4640-a2df-d10c4a933ba0")
        super().__init__(channel_id)
        self.parameters = {}
        self.has_sent_parameters = False
    
    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Called when a message is received from Unity.
        
        Args:
            msg: The incoming message from Unity.
        """
        # Parse response from Unity (e.g., confirmation or current parameter values)
        message = msg.read_string()
        
        if message.startswith("CURRENT_PARAMETERS:"):
            try:
                params_json = message[len("CURRENT_PARAMETERS:"):]
                received_params = json.loads(params_json)
                print(f"Current game parameters: {received_params}")
            except json.JSONDecodeError:
                print(f"Error parsing parameters JSON: {params_json}")
    
    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set a single game parameter.
        
        Args:
            key: The parameter key
            value: The parameter value
        """
        self.parameters[key] = value
        self._send_parameters()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set multiple game parameters at once.
        
        Args:
            parameters: Dictionary of parameter key-value pairs
        """
        self.parameters.update(parameters)
        self._send_parameters()
    
    def _send_parameters(self) -> None:
        """
        Send the current parameters to Unity.
        """
        # Create an outgoing message with JSON-encoded parameters
        outgoing_msg = OutgoingMessage()
        params_json = json.dumps(self.parameters)
        outgoing_msg.write_string(f"SET_PARAMETERS:{params_json}")
        self.queue_message_to_send(outgoing_msg)
        self.has_sent_parameters = True
    
    def request_current_parameters(self) -> None:
        """
        Request the current parameter values from Unity.
        """
        outgoing_msg = OutgoingMessage()
        outgoing_msg.write_string("GET_PARAMETERS")
        self.queue_message_to_send(outgoing_msg)
