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
    by MLGame3D and in which mode (manual or AI).
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
    
    def set_controlled_players(self, player_ids: List[int], control_modes: List[str] = None, player_names: List[str] = None) -> None:
        """
        Set which players should be controlled by MLGame3D and in which mode.
        
        Args:
            player_ids: A list of player IDs to control (0-3 for P1-P4).
            control_modes: A list of control modes ("manual" or "mlplay") for each player.
                           If None, all players are assumed to be in "mlplay" mode.
            player_names: A list of player names to display in Unity. If None or a specific element is None,
                          default names will be used.
        """
        # Map player_ids (0-3) to PlayerID enum in Unity (P1-P4)
        player_enums = [f"P{player_id + 1}" for player_id in player_ids]
        
        # If control_modes is not provided, default to "mlplay" for all players
        if control_modes is None:
            control_modes = ["mlplay"] * len(player_ids)
        
        # Ensure control_modes has the same length as player_ids
        if len(control_modes) != len(player_ids):
            raise ValueError("control_modes must have the same length as player_ids")
        
        # If player_names is not provided, use None for all players
        if player_names is None:
            player_names = [None] * len(player_ids)
        
        # Ensure player_names has the same length as player_ids
        if len(player_names) != len(player_ids):
            raise ValueError("player_names must have the same length as player_ids")
        
        # Create player info strings with format "P1:manual:name", "P2:mlplay:name", etc.
        # If a player name is None, omit the name part
        player_info = []
        for i in range(len(player_ids)):
            if player_names[i] is None:
                player_info.append(f"{player_enums[i]}:{control_modes[i]}")
            else:
                player_info.append(f"{player_enums[i]}:{control_modes[i]}:{player_names[i]}")
        
        # Create an outgoing message
        outgoing_msg = OutgoingMessage()
        outgoing_msg.write_string(f"CONTROL_PLAYERS:{','.join(player_info)}")
        self.queue_message_to_send(outgoing_msg)
        self.has_sent_control_message = True
        
    def set_decision_period(self, decision_period: int) -> None:
        """
        Set the decision period for all controlled players.
        
        Args:
            decision_period: The number of FixedUpdate steps between decisions.
                            Should be a multiple of 20 ms.
        """
        # Create an outgoing message
        outgoing_msg = OutgoingMessage()
        outgoing_msg.write_string(f"SET_DECISION_PERIOD:{decision_period}")
        self.queue_message_to_send(outgoing_msg)