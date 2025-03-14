"""
Game Runner Module

This module provides a class for running games with agents in Unity environments.
"""

import time
import numpy as np
from typing import Dict, Any, List

from mlgame3d.game_env import GameEnvironment
from mlgame3d.agent import Agent

class GameRunner:
    """
    A class for running games with agents in Unity environments.
    """
    
    def __init__(
        self, 
        env: GameEnvironment, 
        agents: List[Agent],
        max_episodes: int = 10,
        max_steps_per_episode: int = 1000,
        render: bool = True,
        render_fps: int = 30
    ):
        """
        Initialize the game runner.
        
        Args:
            env: The game environment
            agents: A list of agents to use (up to 4)
            max_episodes: The maximum number of episodes to run
            max_steps_per_episode: The maximum number of steps per episode
            render: Whether to render the game
            render_fps: The frames per second for rendering
        """
        self.env = env
        
        # Ensure the number of agents matches the environment's configuration
        if len(agents) != env.num_agents:
            raise ValueError(f"Number of agents ({len(agents)}) does not match environment configuration ({env.num_agents})")
            
        self.agents = agents
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.render = render
        self.render_fps = render_fps
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.agent_rewards = [[] for _ in range(len(agents))]
    
    def run(self) -> Dict[str, Any]:
        """
        Run the game for the specified number of episodes.
        
        Returns:
            A dictionary of statistics about the run
        """
        for episode in range(self.max_episodes):
            # Reset the environment and agents
            observations = self.env.reset()
            for agent in self.agents:
                agent.reset()
            
            episode_rewards = [0.0] * len(self.agents)
            episode_step = 0
            done = False
            
            print(f"Starting episode {episode+1}/{self.max_episodes}")
            
            # Run the episode
            while not done and episode_step < self.max_steps_per_episode:
                # Get actions from all agents
                actions = [agent.act(obs) for agent, obs in zip(self.agents, observations)]
                
                # Take a step in the environment
                next_observations, rewards, done, info = self.env.step(actions)
                
                # Let each agent observe the result
                for i, agent in enumerate(self.agents):
                    agent.observe(next_observations[i], rewards[i], done, info)
                    episode_rewards[i] += rewards[i]
                
                # Update for the next step
                observations = next_observations
                episode_step += 1
                
                # Control the rendering frame rate
                if self.render:
                    time.sleep(1.0 / self.render_fps)
            
            # Record episode statistics
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_steps.append(episode_step)
            
            for i, reward in enumerate(episode_rewards):
                self.agent_rewards[i].append(reward)
            
            print(f"Episode {episode+1} finished: total_reward={sum(episode_rewards):.2f}, steps={episode_step}")
            for i, reward in enumerate(episode_rewards):
                print(f"  Agent {i+1} ({self.agents[i].name}): reward={reward:.2f}")
        
        # Return statistics
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the run.
        
        Returns:
            A dictionary of statistics
        """
        stats = {
            "agent_stats": [agent.get_stats() for agent in self.agents],
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "max_reward": np.max(self.episode_rewards) if self.episode_rewards else 0.0,
            "min_reward": np.min(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_steps": np.mean(self.episode_steps) if self.episode_steps else 0.0,
            "agent_rewards": self.agent_rewards,
            "agent_mean_rewards": [np.mean(rewards) if rewards else 0.0 for rewards in self.agent_rewards]
        }
        
        return stats
