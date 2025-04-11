"""
Game Runner Module

This module provides a class for running games with MLPlay instances in Unity environments asynchronously.
"""

import time
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
        render_fps: int = 30,
        mlplay_timeout: float = 0.1,  # Default timeout for MLPlay actions
        game_parameters: Dict[str, Any] = None  # Game parameters to pass to MLPlay instances
    ):
        """
        Initialize the game runner.
        
        Args:
            env: The game environment
            mlplays: A list of MLPlay instances to use (up to 4)
            max_episodes: The maximum number of episodes to run
            max_steps_per_episode: The maximum number of steps per episode
            render: Whether to render the game
            render_fps: The frames per second for rendering
            mlplay_timeout: Timeout in seconds for MLPlay actions
        """
        self.env = env
        
        # Ensure the number of MLPlay instances matches the environment's configuration
        if len(mlplays) != env.num_agents:
            raise ValueError(f"Number of MLPlay instances ({len(mlplays)}) does not match environment configuration ({env.num_agents})")
            
        self.mlplays = mlplays
        self.max_episodes = max_episodes
        self.render = render
        self.render_fps = render_fps
        self.mlplay_timeout = mlplay_timeout
        self.executor = ThreadPoolExecutor(max_workers=len(mlplays))
        self.game_parameters = game_parameters or {}
        
        # Pass game parameters to MLPlay instances if they have parameters in __init__
        for mlplay in mlplays:
            if hasattr(mlplay, 'parameters'):
                mlplay.parameters.update(self.game_parameters)
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.mlplay_rewards = [[] for _ in range(len(mlplays))]
        self.mlplay_names = [getattr(mlplay, 'name', f"MLPlay{i+1}") for i, mlplay in enumerate(mlplays)]
        self.mlplay_total_rewards = [0.0 for _ in range(len(mlplays))]
        self.mlplay_episode_counts = [0 for _ in range(len(mlplays))]
    
    def run(self) -> Dict[str, Any]:
        """
        Run the game for the specified number of episodes.
        
        Returns:
            A dictionary of statistics about the run
        """
        for episode in range(self.max_episodes):
            # Reset the environment and MLPlay instances
            observations = self.env.reset()

            episode_rewards = [0.0] * len(self.mlplays)
            episode_step = 0
            done = False
            
            print(f"Starting episode {episode+1}/{self.max_episodes}")
            
            # Run the episode
            while not done:
                # Start timing the step
                step_start_time = time.time()
                
                # Get actions from all MLPlay instances asynchronously
                actions = self._update_mlplays_async(observations, [0.0] * len(self.mlplays), done, {})
                
                # Take a step in the environment
                next_observations, rewards, done, info = self.env.step(actions)
                
                # Update for the next step
                observations = next_observations
                
                # Update rewards
                for i, reward in enumerate(rewards):
                    episode_rewards[i] += reward
                
                episode_step += 1
                
                # Calculate how much time to sleep to maintain the desired frame rate
                step_elapsed_time = time.time() - step_start_time
                sleep_time = max(0, (1.0 / self.render_fps) - step_elapsed_time)
                
                # Control the rendering frame rate
                if self.render and sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Record episode statistics
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_steps.append(episode_step)
            
            for i, reward in enumerate(episode_rewards):
                self.mlplay_rewards[i].append(reward)
                self.mlplay_total_rewards[i] += reward
                self.mlplay_episode_counts[i] += 1
            
            print(f"Episode {episode+1} finished: total_reward={sum(episode_rewards):.2f}, steps={episode_step}")
            for i, reward in enumerate(episode_rewards):
                print(f"  MLPlay {i+1} ({self.mlplay_names[i]}): reward={reward:.2f}")

            for mlplay in self.mlplays:
                if hasattr(mlplay, 'reset') and callable(getattr(mlplay, 'reset')):
                    mlplay.reset()
        
        # Return statistics
        return self.get_stats()
    
    def _update_mlplays_async(self, 
                            observations: List[Dict[str, np.ndarray]],
                            rewards: List[float],
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
                future = self.executor.submit(
                    mlplay.update, 
                    observations[i], 
                    rewards[i], 
                    done, 
                    info
                )
                futures.append((future, i))
            else:
                print(f"Warning: MLPlay instance {i+1} does not have an update method.")
        
        # Wait for all futures to complete with timeout
        actions = [None] * len(self.mlplays)  # Initialize with None
        for future, i in futures:
            try:
                # Wait for the MLPlay instance to update with timeout
                action = future.result(timeout=self.mlplay_timeout)
                actions[i] = action
            except TimeoutError:
                print(f"MLPlay {self.mlplay_names[i]} timed out after {self.mlplay_timeout:.3f}s. Using default action.")
                # Cancel the future to prevent it from continuing to run in the background
                future.cancel()
                # Use a default action if the MLPlay instance times out
                action_spec = self.env.get_action_space_info()
                if action_spec.is_continuous():
                    actions[i] = np.zeros(action_spec.continuous_size)
                elif action_spec.is_discrete():
                    actions[i] = np.zeros(action_spec.discrete_size, dtype=np.int32)
                else:
                    # Hybrid action space
                    actions[i] = (
                        np.zeros(action_spec.continuous_size),
                        np.zeros(action_spec.discrete_size, dtype=np.int32)
                    )
            except Exception as e:
                print(f"Error updating MLPlay {self.mlplay_names[i]}: {e}")
                # Use a default action if the MLPlay instance fails
                action_spec = self.env.get_action_space_info()
                if action_spec.is_continuous():
                    actions[i] = np.zeros(action_spec.continuous_size)
                elif action_spec.is_discrete():
                    actions[i] = np.zeros(action_spec.discrete_size, dtype=np.int32)
                else:
                    # Hybrid action space
                    actions[i] = (
                        np.zeros(action_spec.continuous_size),
                        np.zeros(action_spec.discrete_size, dtype=np.int32)
                    )
        
        return actions
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the run.
        
        Returns:
            A dictionary of statistics
        """
        mlplay_stats = []
        for i, name in enumerate(self.mlplay_names):
            mlplay_stats.append({
                "name": name,
                "total_reward": self.mlplay_total_rewards[i],
                "episode_count": self.mlplay_episode_counts[i],
                "average_reward": self.mlplay_total_rewards[i] / max(1, self.mlplay_episode_counts[i]),
            })
        
        stats = {
            "mlplay_stats": mlplay_stats,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "max_reward": np.max(self.episode_rewards) if self.episode_rewards else 0.0,
            "min_reward": np.min(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_steps": np.mean(self.episode_steps) if self.episode_steps else 0.0,
            "mlplay_rewards": self.mlplay_rewards,
            "mlplay_mean_rewards": [np.mean(rewards) if rewards else 0.0 for rewards in self.mlplay_rewards]
        }
        
        return stats
