"""
Game Runner Module

This module provides a class for running games with agents in Unity environments.
"""

import time
import numpy as np
from typing import Dict, Any

from mlgame3d.game_env import GameEnvironment
from mlgame3d.agent import Agent

class GameRunner:
    """
    A class for running games with agents in Unity environments.
    """
    
    def __init__(
        self, 
        env: GameEnvironment, 
        agent: Agent,
        max_episodes: int = 10,
        max_steps_per_episode: int = 1000,
        render: bool = True,
        render_fps: int = 30
    ):
        """
        Initialize the game runner.
        
        Args:
            env: The game environment
            agent: The agent to use
            max_episodes: The maximum number of episodes to run
            max_steps_per_episode: The maximum number of steps per episode
            render: Whether to render the game
            render_fps: The frames per second for rendering
        """
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.render = render
        self.render_fps = render_fps
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
    
    def run(self) -> Dict[str, Any]:
        """
        Run the game for the specified number of episodes.
        
        Returns:
            A dictionary of statistics about the run
        """
        for episode in range(self.max_episodes):
            # Reset the environment and agent
            observations = self.env.reset()
            self.agent.reset()
            
            episode_reward = 0.0
            episode_step = 0
            done = False
            
            print(f"Starting episode {episode+1}/{self.max_episodes}")
            
            # Run the episode
            while not done and episode_step < self.max_steps_per_episode:
                # Get the action from the agent
                action = self.agent.act(observations)
                
                # Take a step in the environment
                next_observations, reward, done, info = self.env.step(action)
                
                # Let the agent observe the result
                self.agent.observe(next_observations, reward, done, info)
                
                # Update for the next step
                observations = next_observations
                episode_reward += reward
                episode_step += 1
                
                # Control the rendering frame rate
                if self.render:
                    time.sleep(1.0 / self.render_fps)
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(episode_step)
            
            print(f"Episode {episode+1} finished: reward={episode_reward:.2f}, steps={episode_step}")
        
        # Return statistics
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the run.
        
        Returns:
            A dictionary of statistics
        """
        return {
            "agent_stats": self.agent.get_stats(),
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "max_reward": np.max(self.episode_rewards) if self.episode_rewards else 0.0,
            "min_reward": np.min(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_steps": np.mean(self.episode_steps) if self.episode_steps else 0.0
        }
