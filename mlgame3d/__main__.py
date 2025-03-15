"""
MLGame3D Main Module

This module provides the command-line interface for the MLGame3D framework.
"""

import argparse
import sys
import os
from typing import List, Optional

from mlgame3d import __version__
from mlgame3d.game_env import GameEnvironment
from mlgame3d.mlplay import RandomMLPlay
from mlgame3d.game_runner import GameRunner
from mlgame3d.mlplay_loader import create_mlplay_from_file, validate_mlplay_file
from mlagents_envs.exception import UnityCommunicatorStoppedException

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
        "--max-steps", "-ms",
        type=int, 
        default=1000, 
        help="Maximum number of steps per episode"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=int, 
        default=30, 
        help="Frames per second for rendering"
    )
    
    parser.add_argument(
        "--num-agents", "-na",
        type=int, 
        default=1, 
        help="Number of MLPlay instances to use (up to 4)"
    )
    
    parser.add_argument(
        "--mlplay1", "-m1",
        type=str, 
        default=None, 
        help="Path to a Python file containing an MLPlay class for instance 1. If not provided, a RandomMLPlay will be used."
    )
    
    parser.add_argument(
        "--mlplay2", "-m2",
        type=str, 
        default=None, 
        help="Path to a Python file containing an MLPlay class for instance 2. If not provided, a RandomMLPlay will be used."
    )
    
    parser.add_argument(
        "--mlplay3", "-m3",
        type=str, 
        default=None, 
        help="Path to a Python file containing an MLPlay class for instance 3. If not provided, a RandomMLPlay will be used."
    )
    
    parser.add_argument(
        "--mlplay4", "-m4",
        type=str, 
        default=None, 
        help="Path to a Python file containing an MLPlay class for instance 4. If not provided, a RandomMLPlay will be used."
    )
    
    parser.add_argument(
        "game_executable", 
        nargs="?", 
        default=None, 
        help="Path to the Unity game executable. If None, will connect to an already running Unity editor"
    )
    
    parser.add_argument(
        "game_params", 
        nargs="*", 
        help="Additional parameters to pass to the Unity game"
    )
    
    return parser.parse_args(args)

def validate_mlplay_args(parsed_args):
    """
    Validate MLPlay-related command-line arguments.
    
    Args:
        parsed_args: Parsed command-line arguments.
        
    Raises:
        ValueError: If the arguments are invalid.
    """
    # Check if the number of MLPlay instances is valid
    if parsed_args.num_agents < 1 or parsed_args.num_agents > 4:
        raise ValueError(f"Number of MLPlay instances must be between 1 and 4, got {parsed_args.num_agents}")
        
    # Check if the MLPlay files exist and are valid
    mlplay_files = [
        parsed_args.mlplay1,
        parsed_args.mlplay2,
        parsed_args.mlplay3,
        parsed_args.mlplay4
    ][:parsed_args.num_agents]
    
    for i, file_path in enumerate(mlplay_files):
        if file_path is not None:
            if not os.path.exists(file_path):
                raise ValueError(f"MLPlay file not found: {file_path}")
                
            if not validate_mlplay_file(file_path):
                raise ValueError(f"Invalid MLPlay file: {file_path}")

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the MLGame3D framework.
    
    Args:
        args: Command-line arguments. If None, sys.argv[1:] is used.
        
    Returns:
        Exit code.
    """
    parsed_args = parse_args(args)
    
    try:
        # Validate MLPlay-related arguments
        validate_mlplay_args(parsed_args)
        
        # Create the environment
        env = GameEnvironment(
            file_name=parsed_args.game_executable,
            worker_id=parsed_args.worker_id,
            base_port=parsed_args.base_port,
            seed=parsed_args.seed,
            no_graphics=parsed_args.no_graphics,
            timeout_wait=parsed_args.timeout,
            num_agents=parsed_args.num_agents
        )
        
        try:
            # Get information about the action space
            action_space_info = env.get_action_space_info()
            
            # Create MLPlay instances
            mlplays = []
            mlplay_files = [
                parsed_args.mlplay1,
                parsed_args.mlplay2,
                parsed_args.mlplay3,
                parsed_args.mlplay4
            ][:parsed_args.num_agents]
            
            for i, file_path in enumerate(mlplay_files):
                if file_path is not None:
                    try:
                        mlplay = create_mlplay_from_file(file_path, action_space_info, name=f"MLPlay{i+1}")
                        mlplays.append(mlplay)
                    except Exception as e:
                        print(f"Error creating MLPlay instance from file {file_path}: {e}")
                        print(f"Using RandomMLPlay for instance {i+1} instead.")
                        mlplays.append(RandomMLPlay(action_space_info, name=f"RandomMLPlay{i+1}"))
                else:
                    mlplays.append(RandomMLPlay(action_space_info, name=f"RandomMLPlay{i+1}"))
            
            # Create a game runner
            # Calculate MLPlay timeout based on fps
            # Use 80% of the frame time as the timeout to ensure the game loop runs smoothly
            mlplay_timeout = 0.8 / parsed_args.fps if parsed_args.fps > 0 else 0.1
            
            runner = GameRunner(
                env=env,
                mlplays=mlplays,
                max_episodes=parsed_args.episodes,
                max_steps_per_episode=parsed_args.max_steps,
                render=not parsed_args.no_graphics,
                render_fps=parsed_args.fps,
                mlplay_timeout=mlplay_timeout
            )
            
            # Run the game
            stats = runner.run()
            
            # Print statistics
            print("\nRun Statistics:")
            print(f"Mean Total Reward: {stats['mean_reward']:.2f}")
            print(f"Max Total Reward: {stats['max_reward']:.2f}")
            print(f"Min Total Reward: {stats['min_reward']:.2f}")
            print(f"Mean Steps: {stats['mean_steps']:.2f}")
            
            print("\nMLPlay Statistics:")
            for i, mlplay_mean_reward in enumerate(stats['mlplay_mean_rewards']):
                print(f"  MLPlay {i+1} ({mlplays[i].name}): Mean Reward: {mlplay_mean_reward:.2f}")
            
            return 0
        
        except UnityCommunicatorStoppedException:
            print("Unity environment stopped.")
            return 1
            
        finally:
            # Make sure to close the environment
            env.close()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
