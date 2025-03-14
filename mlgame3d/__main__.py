"""
MLGame3D Main Module

This module provides the command-line interface for the MLGame3D framework.
"""

import argparse
import sys
from typing import List, Optional

from mlgame3d import __version__
from mlgame3d.game_env import GameEnvironment
from mlgame3d.agent import RandomAgent
from mlgame3d.game_runner import GameRunner
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
        description="MLGame3D - A framework for playing Unity games with Python agents using ML-Agents",
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
        # Create the environment
        env = GameEnvironment(
            file_name=parsed_args.game_executable,
            worker_id=parsed_args.worker_id,
            base_port=parsed_args.base_port,
            seed=parsed_args.seed,
            no_graphics=parsed_args.no_graphics,
            timeout_wait=parsed_args.timeout
        )
        
        try:
            # Get information about the action space
            action_space_info = env.get_action_space_info()
            
            # Create a random agent
            agent = RandomAgent(action_space_info)
            
            # Create a game runner
            runner = GameRunner(
                env=env,
                agent=agent,
                max_episodes=parsed_args.episodes,
                max_steps_per_episode=parsed_args.max_steps,
                render=not parsed_args.no_graphics,
                render_fps=parsed_args.fps
            )
            
            # Run the game
            stats = runner.run()
            
            # Print statistics
            print("\nRun Statistics:")
            print(f"Mean Reward: {stats['mean_reward']:.2f}")
            print(f"Max Reward: {stats['max_reward']:.2f}")
            print(f"Min Reward: {stats['min_reward']:.2f}")
            print(f"Mean Steps: {stats['mean_steps']:.2f}")
            
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
