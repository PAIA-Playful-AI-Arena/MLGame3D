"""
Player Control Side Channel Module

This module provides a side channel for sending player control information to Unity.
"""

import uuid
from typing import List
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage

class PlayerControlSideChannel(SideChannel):
    """
    A side channel for sending player control information to Unity.
    
    This side channel sends information about which players should be controlled
    by MLGame3D.
    """
    
    def __init__(self) -> None:
        """
        Initialize the player control side channel.
        """
        # Use a unique channel ID
        channel_id = uuid.UUID("b47f2099-7be5-4dbc-8b3d-d3f2c4b1a95e")
        super().__init__(channel_id)
        self.has_sent_control_message = False
    
    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Called when a message is received from Unity.
        
        Args:
            msg: The incoming message from Unity.
        """
        # Currently, we don't expect any messages from Unity
        pass
    
    def set_controlled_players(self, player_ids: List[int]) -> None:
        """
        Set which players should be controlled by MLGame3D.
        
        Args:
            player_ids: A list of player IDs to control (0-3 for P1-P4).
        """
        # Map player_ids (0-3) to PlayerID enum in Unity (P1-P4)
        player_enums = [f"P{player_id + 1}" for player_id in player_ids]
        
        # Create an outgoing message
        outgoing_msg = OutgoingMessage()
        outgoing_msg.write_string(f"CONTROL_PLAYERS:{','.join(player_enums)}")
        self.queue_message_to_send(outgoing_msg)
        self.has_sent_control_message = True
        print(f"Sent control message to Unity: CONTROL_PLAYERS:{','.join(player_enums)}")