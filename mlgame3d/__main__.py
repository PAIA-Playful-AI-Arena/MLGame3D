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
from mlgame3d.agent import RandomAgent
from mlgame3d.game_runner import GameRunner
from mlgame3d.agent_loader import create_agent_from_file, validate_agent_file
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
        "--num-agents", "-na",
        type=int, 
        default=1, 
        help="Number of agents to use (up to 4)"
    )
    
    parser.add_argument(
        "--agent1", "-a1",
        type=str, 
        default=None, 
        help="Path to a Python file containing an Agent class for agent 1. If not provided, a RandomAgent will be used."
    )
    
    parser.add_argument(
        "--agent2", "-a2",
        type=str, 
        default=None, 
        help="Path to a Python file containing an Agent class for agent 2. If not provided, a RandomAgent will be used."
    )
    
    parser.add_argument(
        "--agent3", "-a3",
        type=str, 
        default=None, 
        help="Path to a Python file containing an Agent class for agent 3. If not provided, a RandomAgent will be used."
    )
    
    parser.add_argument(
        "--agent4", "-a4",
        type=str, 
        default=None, 
        help="Path to a Python file containing an Agent class for agent 4. If not provided, a RandomAgent will be used."
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

def validate_agent_args(parsed_args):
    """
    Validate agent-related command-line arguments.
    
    Args:
        parsed_args: Parsed command-line arguments.
        
    Raises:
        ValueError: If the arguments are invalid.
    """
    # Check if the number of agents is valid
    if parsed_args.num_agents < 1 or parsed_args.num_agents > 4:
        raise ValueError(f"Number of agents must be between 1 and 4, got {parsed_args.num_agents}")
        
    # Check if the agent files exist and are valid
    agent_files = [
        parsed_args.agent1,
        parsed_args.agent2,
        parsed_args.agent3,
        parsed_args.agent4
    ][:parsed_args.num_agents]
    
    for i, file_path in enumerate(agent_files):
        if file_path is not None:
            if not os.path.exists(file_path):
                raise ValueError(f"Agent file not found: {file_path}")
                
            if not validate_agent_file(file_path):
                raise ValueError(f"Invalid agent file: {file_path}")

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
        # Validate agent-related arguments
        validate_agent_args(parsed_args)
        
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
            
            # Create agents
            agents = []
            agent_files = [
                parsed_args.agent1,
                parsed_args.agent2,
                parsed_args.agent3,
                parsed_args.agent4
            ][:parsed_args.num_agents]
            
            for i, file_path in enumerate(agent_files):
                if file_path is not None:
                    try:
                        agent = create_agent_from_file(file_path, action_space_info, name=f"Agent{i+1}")
                        agents.append(agent)
                    except Exception as e:
                        print(f"Error creating agent from file {file_path}: {e}")
                        print(f"Using RandomAgent for agent {i+1} instead.")
                        agents.append(RandomAgent(action_space_info, name=f"RandomAgent{i+1}"))
                else:
                    agents.append(RandomAgent(action_space_info, name=f"RandomAgent{i+1}"))
            
            # Create a game runner
            runner = GameRunner(
                env=env,
                agents=agents,
                max_episodes=parsed_args.episodes,
                max_steps_per_episode=parsed_args.max_steps,
                render=not parsed_args.no_graphics,
                render_fps=parsed_args.fps
            )
            
            # Run the game
            stats = runner.run()
            
            # Print statistics
            print("\nRun Statistics:")
            print(f"Mean Total Reward: {stats['mean_reward']:.2f}")
            print(f"Max Total Reward: {stats['max_reward']:.2f}")
            print(f"Min Total Reward: {stats['min_reward']:.2f}")
            print(f"Mean Steps: {stats['mean_steps']:.2f}")
            
            print("\nAgent Statistics:")
            for i, agent_mean_reward in enumerate(stats['agent_mean_rewards']):
                print(f"  Agent {i+1} ({agents[i].name}): Mean Reward: {agent_mean_reward:.2f}")
            
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
