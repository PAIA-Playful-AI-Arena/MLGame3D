"""
Ranking Side Channel Module

This module provides a side channel for receiving ranking data from Unity at the end of each episode.
"""

import uuid
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import os
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage

class RankingSideChannel(SideChannel):
    """
    A side channel for receiving ranking data from Unity.
    
    This side channel receives the ranking data from Unity at the end of each episode
    and provides methods for accessing and analyzing that data.
    """
    
    def __init__(self, result_output_file: Optional[str] = None) -> None:
        """
        Initialize the ranking side channel.
        
        Args:
            result_output_file: Optional path to a CSV file where result data will be saved.
                               If provided, each episode's result data will be appended to this file.
                               If the file path doesn't end with '.csv', it will be automatically added.
        """
        # Use the same channel ID as in the Unity side
        channel_id = uuid.UUID("7a8e3f52-9c5d-4b6a-8f1e-d3b9c0a7e4d5")
        super().__init__(channel_id)
        self.ranking_data = None
        self.has_requested_data = False
        self.episode_rankings = []
        
        # Ensure file path ends with .csv if provided
        if result_output_file is not None:
            if not result_output_file.lower().endswith('.csv'):
                result_output_file += '.csv'
        
        self.result_output_file = result_output_file
        self.episode_count = 0
    
    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Called when a message is received from Unity.
        
        Args:
            msg: The incoming message from Unity.
        """
        # Read the message as a string
        message = msg.read_string()
        
        if message.startswith("RANKING_DATA:"):
            try:
                # Parse the JSON string
                json_str = message[len("RANKING_DATA:"):]
                self.ranking_data = json.loads(json_str)
                
                # Store the ranking data for this episode
                self.episode_rankings.append(self.ranking_data)
                
                # Increment episode count
                self.episode_count += 1
                
                # Print the ranking table
                self.print_ranking_table()
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {json_str}")
    
    def request_ranking_data(self) -> None:
        """
        Request the ranking data from Unity.
        """
        if not self.has_requested_data:
            # Create an outgoing message
            outgoing_msg = OutgoingMessage()
            outgoing_msg.write_string("REQUEST_RANKING_DATA")
            self.queue_message_to_send(outgoing_msg)
            self.has_requested_data = True
            print("Requested ranking data from Unity")
    
    def has_ranking_data(self) -> bool:
        """
        Check if the ranking data has been received.
        
        Returns:
            True if the ranking data has been received, False otherwise.
        """
        return self.ranking_data is not None
    
    def get_latest_ranking_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest ranking data received from Unity.
        
        Returns:
            The latest ranking data or None if no data has been received.
        """
        return self.ranking_data
    
    def get_all_episode_rankings(self) -> List[Dict[str, Any]]:
        """
        Get the ranking data for all episodes.
        
        Returns:
            A list of ranking data for each episode.
        """
        return self.episode_rankings
    
    def get_player_rankings(self, player_id: int = None) -> List[Dict[str, Any]]:
        """
        Get the rankings for a specific player or all players.
        
        Args:
            player_id: The ID of the player to get rankings for, or None for all players.
            
        Returns:
            A list of player rankings.
        """
        if not self.has_ranking_data():
            return []
        
        rankings = self.ranking_data.get("rankings", [])
        
        if player_id is not None:
            # Filter rankings for the specified player
            return [r for r in rankings if int(r.get("player_num", "0P")[0]) - 1 == player_id]
        else:
            return rankings
    
    def print_ranking_table(self) -> None:
        """
        Print a ranking table using pandas DataFrame.
        The table is sorted by player_num.
        If result_output_file is set, also save the result data to the file.
        """
        if not self.has_ranking_data():
            print("No ranking data available.")
            return
        
        rankings = self.ranking_data.get("rankings", [])
        if not rankings:
            print("No player rankings available.")
            return
        
        # Convert rankings to pandas DataFrame
        df = pd.DataFrame(rankings)
        
        # Sort by player_num
        if 'player_num' in df.columns:
            # Extract numeric part from player_num for sorting (e.g., "1P" -> 1)
            df['sort_key'] = df['player_num'].str.extract(r'(\d+)').astype(int)
            df = df.sort_values('sort_key')
            df = df.drop('sort_key', axis=1)
        
        # Print the DataFrame
        print(df.to_string(index=False))
        
        # Save to file if output file is specified
        if self.result_output_file:
            self.save_result_to_file(df)
    
    def clear_episode_data(self) -> None:
        """
        Clear the ranking data for the current episode.
        """
        self.ranking_data = None
        self.has_requested_data = False
        
    def save_result_to_file(self, df: pd.DataFrame) -> None:
        """
        Save the result data to a CSV file.
        Each episode's data is appended to the file with the episode number included.
        
        Args:
            df: The DataFrame containing the result data.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.result_output_file)), exist_ok=True)
            
            # Add episode information to the DataFrame
            df = df.copy()
            df['episode'] = self.episode_count
            
            # Ensure episode column is the first column
            cols = list(df.columns)
            if 'episode' in cols:
                cols.remove('episode')
                cols = ['episode'] + cols
                df = df[cols]
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.isfile(self.result_output_file)
            
            # Save to CSV file (append mode if file exists)
            df.to_csv(self.result_output_file, mode='a', index=False, header=not file_exists)
            print(f"Result data appended to {self.result_output_file}")
        except Exception as e:
            print(f"Error saving ranking data to file: {e}")
    
    def set_result_output_file(self, file_path: str) -> None:
        """
        Set or update the file path for saving result data.
        
        Args:
            file_path: The path to the CSV file where result data will be saved.
                      If the file path doesn't end with '.csv', it will be automatically added.
        """
        # Ensure file path ends with .csv
        if not file_path.lower().endswith('.csv'):
            file_path += '.csv'
            
        self.result_output_file = file_path