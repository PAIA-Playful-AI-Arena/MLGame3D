"""
MLGame3D Main Module

This module provides the command-line interface for the MLGame3D framework.
"""

import argparse
import sys
import os
import traceback
from typing import List, Optional

from mlgame3d import __version__
from mlgame3d.game_env import GameEnvironment
from mlgame3d.mlplay import RandomMLPlay
from mlgame3d.game_runner import GameRunner
from mlgame3d.mlplay_loader import create_mlplay_from_file, validate_mlplay_file
from mlagents_envs.exception import UnityCommunicatorStoppedException

# Set environment variables to suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_LOG_LEVEL"] = "ERROR"

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments. If None, sys.argv[1:] is used.
        
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="MLGame3D - A framework for playing Unity games with Python MLPlay classes using ML-Agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version", 
        version=f"MLGame3D {__version__}"
    )
    
    parser.add_argument(
        "--no-graphics", "-ng",
        action="store_true", 
        help="Run the Unity simulator in no-graphics mode"
    )
    
    parser.add_argument(
        "--worker-id", "-w",
        type=int, 
        default=0, 
        help="Offset from base port. Used for training multiple environments simultaneously"
    )
    
    parser.add_argument(
        "--base-port", "-p",
        type=int, 
        default=None, 
        help="Base port to connect to Unity environment. If None, defaults to 5004 for editor or 5005 for executable"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int, 
        default=0, 
        help="Random seed for the environment"
    )
    
    parser.add_argument(
        "--timeout", "-t",
        type=int, 
        default=60, 
        help="Time (in seconds) to wait for connection from environment"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int, 
        default=5, 
        help="Number of episodes to run"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=int, 
        default=60, 
        help="Target number of frames per second for rendering"
    )
    
    parser.add_argument(
        "--time-scale", "-ts",
        type=float,
        default=1.0,
        help="Time scale factor for the simulation. Higher values make the simulation run faster. "
             "Note: Values less than 1.0 will be clamped to 1.0 by Unity and have no effect"
    )
    
    parser.add_argument(
        "--decision-period", "-dp",
        type=int,
        default=5,
        help="Number of FixedUpdate steps between AI decisions. Should be a multiple of 20ms. Range: 1-20"
    )
    
    # Add a new argument for auto-numbered AI instances
    parser.add_argument(
        "--ai", "-i",
        action="append",
        type=str,
        help="Control mode for an instance (auto-numbered). Can be specified multiple times. Each occurrence is equivalent to --ai1, --ai2, etc. in order."
    )
    
    parser.add_argument(
        "--ai1", "-i1",
        type=str,
        default="manual",
        help="Control mode for instance 1. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default)."
    )
    
    parser.add_argument(
        "--ai2", "-i2",
        type=str,
        default="manual",
        help="Control mode for instance 2. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default)."
    )
    
    parser.add_argument(
        "--ai3", "-i3",
        type=str,
        default="manual",
        help="Control mode for instance 3. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default)."
    )
    
    parser.add_argument(
        "--ai4", "-i4",
        type=str,
        default="manual",
        help="Control mode for instance 4. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default)."
    )
    
    parser.add_argument(
        "game_executable", 
        nargs="?", 
        default=None, 
        help="Path to the Unity game executable. If None, will connect to an already running Unity editor"
    )
    
    parser.add_argument(
        "--game-param", "-gp",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Game parameter in the format KEY VALUE. Can be specified multiple times for different parameters."
    )
    
    parser.add_argument(
        "--result-output-file", "-o",
        type=str,
        default=None,
        help="Path to a CSV file where result data will be saved. Each episode's result will be appended to this file."
    )
    
    return parser.parse_args(args)

def process_ai_settings(parsed_args):
    """
    Process AI settings from command-line arguments.
    
    This function handles both the traditional --ai1, --ai2, etc. arguments
    and the new auto-numbered --ai argument.
    
    Args:
        parsed_args: Parsed command-line arguments.
        
    Returns:
        Tuple of (List of AI settings, List of AI names).
    """
    # Start with the default AI settings
    ai_settings = [
        parsed_args.ai1,
        parsed_args.ai2,
        parsed_args.ai3,
        parsed_args.ai4
    ]
    
    # Initialize AI names list (None means use default name)
    ai_names = [None, None, None, None]
    
    # Apply auto-numbered AI arguments if provided
    if parsed_args.ai:
        for i, ai_arg in enumerate(parsed_args.ai):
            if i < 4:  # We only support up to 4 AI instances
                # Check if AI name is provided (format: "mlplay_file,AI_name" or "manual,AI_name")
                if "," in ai_arg:
                    parts = ai_arg.split(",", 1)  # Split only on the first comma
                    ai_settings[i] = parts[0]
                    ai_names[i] = parts[1] if parts[1] else None
                else:
                    ai_settings[i] = ai_arg
    
    # Process traditional ai1, ai2, etc. arguments for AI names
    for i, setting in enumerate([parsed_args.ai1, parsed_args.ai2, parsed_args.ai3, parsed_args.ai4]):
        if setting and "," in setting:
            parts = setting.split(",", 1)  # Split only on the first comma
            ai_settings[i] = parts[0]
            ai_names[i] = parts[1] if parts[1] else None
    
    return ai_settings, ai_names

def validate_mlplay_args(parsed_args):
    """
    Validate MLPlay-related command-line arguments.
    
    Args:
        parsed_args: Parsed command-line arguments.
        
    Raises:
        ValueError: If the arguments are invalid.
    """
    # Process AI settings
    ai_settings, _ = process_ai_settings(parsed_args)
    
    # Check if the AI files exist and are valid
    for i, setting in enumerate(ai_settings):
        if setting not in ["hidden", "manual"] and setting is not None:
            if not os.path.exists(setting):
                raise ValueError(f"MLPlay file not found: {setting}")
                
            if not validate_mlplay_file(setting):
                raise ValueError(f"Invalid MLPlay file: {setting}")

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the MLGame3D framework.
    
    Args:
        args: Command-line arguments. If None, sys.argv[1:] is used.
        
    Returns:
        Exit code.
    """
    parsed_args = parse_args(args)

    # Process AI settings
    ai_settings, ai_names = process_ai_settings(parsed_args)
    
    try:
        # Validate MLPlay-related arguments
        validate_mlplay_args(parsed_args)

        controlled_players = []
        control_modes = []
        player_names = []
        
        for i, setting in enumerate(ai_settings):
            if setting is not None and setting != "hidden":
                controlled_players.append(i)
                # Set control mode: "mlplay" for Python files, "manual" for manual control
                mode = "manual" if setting == "manual" else "mlplay"
                control_modes.append(mode)
                player_names.append(ai_names[i])
        
        # Create the environment
        env = GameEnvironment(
            file_name=parsed_args.game_executable,
            worker_id=parsed_args.worker_id,
            base_port=parsed_args.base_port,
            seed=parsed_args.seed,
            time_scale=parsed_args.time_scale,
            fps=parsed_args.fps,
            no_graphics=parsed_args.no_graphics,
            timeout_wait=parsed_args.timeout,
            controlled_players=controlled_players,
            control_modes=control_modes,
            player_names=player_names,
            decision_period=parsed_args.decision_period,
            game_parameters=parsed_args.game_param,
            result_output_file=parsed_args.result_output_file,
        )
        
        try:
            # Process game parameters if provided
            game_parameters = {}
            if parsed_args.game_param is not None:
                for key, value in parsed_args.game_param:
                    # Try to convert value to appropriate type
                    try:
                        # Try as int
                        game_parameters[key] = int(value)
                    except ValueError:
                        try:
                            # Try as float
                            game_parameters[key] = float(value)
                        except ValueError:
                            # Try as boolean
                            if value.lower() in ('true', 'yes', '1'):
                                game_parameters[key] = True
                            elif value.lower() in ('false', 'no', '0'):
                                game_parameters[key] = False
                            else:
                                # Keep as string
                                game_parameters[key] = value
            
            # Create MLPlay instances
            mlplays = []
            mlplay_to_behavior_map = {}  # Map to track which MLPlay corresponds to which behavior
            
            # First, create a mapping from player index to behavior name
            player_to_behavior_map = {}
            for i, player_idx in enumerate(controlled_players):
                if player_idx < len(env.behavior_names) and control_modes[i] == "mlplay":
                    player_to_behavior_map[player_idx] = env.behavior_names[player_idx]
            
            # Now create MLPlay instances for each AI setting
            mlplay_index = 0
            for i, setting in enumerate(ai_settings):
                if setting not in ["hidden", "manual"] and setting is not None:
                    # Find the corresponding player index
                    player_idx = i
                    
                    # Check if this player is in the controlled players list
                    if player_idx in player_to_behavior_map:
                        behavior_name = player_to_behavior_map[player_idx]
                        action_space_info = env.get_action_space_info(behavior_name)
                        observation_structure = env.get_observation_structure(behavior_name)
                        try:
                            mlplay = create_mlplay_from_file(
                                setting, observation_structure,
                                action_space_info,
                                name=f"P{player_idx+1}",
                                game_parameters=game_parameters
                            )
                            mlplays.append(mlplay)
                            mlplay_to_behavior_map[mlplay_index] = behavior_name
                            mlplay_index += 1
                        except Exception as e:
                            print(f"Error creating MLPlay instance from file {setting}: {e}")
                            traceback.print_exc()
                            print(f"Using RandomMLPlay for player {player_idx+1} instead.")
                            mlplay = RandomMLPlay(action_space_info, name=f"RandomMLPlay{player_idx+1}")
                            mlplays.append(mlplay)
                            mlplay_to_behavior_map[mlplay_index] = behavior_name
                            mlplay_index += 1
                    else:
                        print(f"Warning: Player {player_idx+1} is not in the controlled players list or is not set to mlplay mode.")
            
            # Create a game runner
            # Calculate MLPlay timeout based on 20ms * decision period, adjusted by time scale
            mlplay_timeout = 0.02 * parsed_args.decision_period / parsed_args.time_scale
            
            runner = GameRunner(
                env=env,
                mlplays=mlplays,
                max_episodes=parsed_args.episodes,
                render=not parsed_args.no_graphics,
                mlplay_timeout=mlplay_timeout,
                game_parameters=game_parameters,
                mlplay_to_behavior_map=mlplay_to_behavior_map
            )
            
            # Run the game
            runner.run()

            return 0
        
        except UnityCommunicatorStoppedException:
            print("Unity environment stopped.")
            return 0
            
        finally:
            # Make sure to close the environment
            env.close()
    
    except Exception as e:
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
