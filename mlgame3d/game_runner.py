"""
Game Runner Module

This module provides a class for running games with MLPlay instances in Unity environments asynchronously.
"""

import traceback
import numpy as np
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from mlgame3d.game_env import GameEnvironment

class GameRunner:
    """
    A class for running games with MLPlay instances in Unity environments asynchronously.
    
    This class supports asynchronous MLPlay execution, which prevents slow MLPlay instances 
    from blocking the game loop.
    """
    
    def __init__(
        self,
        env: GameEnvironment,
        mlplays: List[Any],
        max_episodes: int = 10,
        render: bool = True,
        mlplay_timeout: float = 0.1,  # Default timeout for MLPlay actions
        game_parameters: Dict[str, Any] = None,  # Game parameters to pass to MLPlay instances
        mlplay_to_behavior_map: Dict[int, str] = None  # Mapping from MLPlay index to behavior name
    ):
        """
        Initialize the game runner.
        
        Args:
            env: The game environment
            mlplays: A list of MLPlay instances to use (up to 4)
            max_episodes: The maximum number of episodes to run
            max_steps_per_episode: The maximum number of steps per episode
            render: Whether to render the game
            mlplay_timeout: Timeout in seconds for MLPlay actions
        """
        self.env = env
        self.mlplays = mlplays
        self.max_episodes = max_episodes
        self.render = render
        self.mlplay_timeout = mlplay_timeout
        if len(mlplays) > 0:
            self.executor = ThreadPoolExecutor(max_workers=len(mlplays))
        self.game_parameters = game_parameters or {}
        self.mlplay_to_behavior_map = mlplay_to_behavior_map or {}
        
        # If no mapping is provided, create a default mapping
        if not self.mlplay_to_behavior_map and len(mlplays) > 0:
            for i in range(len(mlplays)):
                if i < len(env.behavior_names):
                    self.mlplay_to_behavior_map[i] = env.behavior_names[i]
        
        # Pass game parameters to MLPlay instances if they have parameters in __init__
        for mlplay in mlplays:
            if hasattr(mlplay, 'parameters'):
                mlplay.parameters.update(self.game_parameters)
        
        # Statistics
        self.mlplay_names = [getattr(mlplay, 'name', f"MLPlay{i+1}") for i, mlplay in enumerate(mlplays)]
    
    def run(self) -> None:
        """
        Run the game for the specified number of episodes.
        
        Returns:
            A dictionary of statistics about the run
        """
        for episode in range(self.max_episodes):
            # Reset the environment and MLPlay instances
            observations = self.env.reset()

            episode_step = 0
            done = False
            info = {}
            
            print(f"Starting episode {episode+1}/{self.max_episodes}")
            
            # Run the episode
            while not done:
                # Get actions from all MLPlay instances asynchronously
                actions = self._update_mlplays_async(observations, done, info)
                
                # Take a step in the environment
                next_observations, rewards, done, info = self.env.step(actions)
                
                # Update for the next step
                observations = next_observations
                
                episode_step += 1
            
            print(f"Episode {episode+1} finished: steps={episode_step}")

            for mlplay in self.mlplays:
                if hasattr(mlplay, 'reset') and callable(getattr(mlplay, 'reset')):
                    try:
                        mlplay.reset()
                    except Exception as e:
                        print(f"Error resetting MLPlay instance {mlplay.name}: {e}")
                        traceback.print_exc()
        
        return
    
    def _update_mlplays_async(self, 
                            observations: List[Dict[str, np.ndarray]],
                            done: bool,
                            info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Update all MLPlay instances asynchronously and get their actions.
        
        Args:
            observations: A list of observations for each MLPlay instance
            rewards: A list of rewards for each MLPlay instance
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            A list of actions from each MLPlay instance
        """
        # Create a future for each MLPlay instance
        futures = []
        for i, mlplay in enumerate(self.mlplays):
            if hasattr(mlplay, 'update') and callable(getattr(mlplay, 'update')):
                # Get the behavior name for this MLPlay instance
                behavior_name = self.mlplay_to_behavior_map.get(i)
                
                if behavior_name is None:
                    print(f"Warning: No behavior name mapped for MLPlay instance {i}. Skipping.")
                    continue
                    
                if behavior_name not in observations:
                    print(f"Warning: Behavior name {behavior_name} not found in observations. Skipping.")
                    continue
                
                # Get keyboard state if available
                keyboard = set()
                if hasattr(self.env, 'keyboard_state_channel'):
                    keyboard = self.env.keyboard_state_channel.get_pressed_keys()
                
                # Check if MLPlay's update method accepts keyboard parameter
                if hasattr(mlplay, 'update'):
                    update_method = getattr(mlplay, 'update')
                    import inspect
                    params = inspect.signature(update_method).parameters
                    
                    if 'keyboard' in params:
                        # MLPlay's update method accepts keyboard parameter
                        future = self.executor.submit(
                            mlplay.update,
                            observations[behavior_name],
                            done,
                            info,
                            keyboard
                        )
                    else:
                        # MLPlay's update method doesn't accept keyboard parameter
                        future = self.executor.submit(
                            mlplay.update,
                            observations[behavior_name],
                            done,
                            info
                        )
                else:
                    continue
                futures.append((future, i))
            else:
                print(f"Warning: MLPlay instance {i+1} does not have an update method.")
        
        # Wait for all futures to complete with timeout
        actions = {}  # Initialize with None
        for future, i in futures:
            # Get the behavior name for this MLPlay instance
            behavior_name = self.mlplay_to_behavior_map.get(i)
            if behavior_name is None:
                continue
                
            try:
                # Wait for the MLPlay instance to update with timeout
                action = future.result(timeout=self.mlplay_timeout)
                # Use a default action if the MLPlay instance returns None
                if action is None:
                    print(f"MLPlay {self.mlplay_names[i]} returned None. Using default action.")
                    action_spec = self.env.get_action_space_info(behavior_name)
                    if action_spec.is_continuous():
                        action = np.zeros(action_spec.continuous_size)
                    elif action_spec.is_discrete():
                        action = np.zeros(action_spec.discrete_size, dtype=np.int32)
                    else:
                        # Hybrid action space
                        action = (
                            np.zeros(action_spec.continuous_size),
                            np.zeros(action_spec.discrete_size, dtype=np.int32)
                        )
                actions[behavior_name] = [action]
            except TimeoutError:
                print(f"MLPlay {self.mlplay_names[i]} timed out after {self.mlplay_timeout:.3f}s. Using default action.")
                # Cancel the future to prevent it from continuing to run in the background
                future.cancel()
                # Use a default action if the MLPlay instance times out
                action_spec = self.env.get_action_space_info(behavior_name)
                if action_spec.is_continuous():
                    actions[behavior_name] = [np.zeros(action_spec.continuous_size)]
                elif action_spec.is_discrete():
                    actions[behavior_name] = [np.zeros(action_spec.discrete_size, dtype=np.int32)]
                else:
                    # Hybrid action space
                    actions[behavior_name] = [(
                        np.zeros(action_spec.continuous_size),
                        np.zeros(action_spec.discrete_size, dtype=np.int32)
                    )]
            except Exception as e:
                print(f"Error updating MLPlay {self.mlplay_names[i]}: {e}")
                traceback.print_exc()
                # Use a default action if the MLPlay instance fails
                action_spec = self.env.get_action_space_info(behavior_name)
                if action_spec.is_continuous():
                    actions[behavior_name] = [np.zeros(action_spec.continuous_size)]
                elif action_spec.is_discrete():
                    actions[behavior_name] = [np.zeros(action_spec.discrete_size, dtype=np.int32)]
                else:
                    # Hybrid action space
                    actions[behavior_name] = [(
                        np.zeros(action_spec.continuous_size),
                        np.zeros(action_spec.discrete_size, dtype=np.int32)
                    )]
        
        return actions
