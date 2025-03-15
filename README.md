# MLGame3D

A framework for playing Unity games with Python agents using ML-Agents for communication with Unity.

## Features

- Simplified communication with Unity games
- Basic agent classes including random agents, keyboard-controlled agents, and human agents
- Game runner for running games and collecting statistics
- Easy to extend and customize
- Command-line interface support
- Support for multiple agents (up to 4) playing simultaneously
- Dynamic loading of external agent files

## Installation

First, make sure you have the ml-agents package installed:

```bash
pip install mlagents
```

Then, you can use this framework directly:

```bash
# Clone the repository
git clone https://github.com/PAIA-Playful-AI-Arena/MLGame3D.git
cd MLGame3D

# Install in development mode (using PEP 517)
pip install -e .
```

If you encounter any deprecation warnings about legacy editable installs, you can use:

```bash
# Install in development mode with PEP 517 explicitly enabled
pip install --use-pep517 -e .
```

## Usage

### Command-line Interface

MLGame3D provides a command-line interface that allows you to launch games directly from the command line:

```bash
python -m mlgame3d [options] <Unity game executable> [game_params]
```

Options include:

- `--version`, `-v`: Display version information
- `--help`, `-h`: Display help information (built-in argparse option)
- `--no-graphics`, `-ng`: Run the Unity simulator in no-graphics mode
- `--worker-id`, `-w`: Set the worker ID for running multiple environments simultaneously (default: 0)
- `--base-port`, `-p`: Set the base port (default: None, will use Unity default port)
- `--seed`, `-s`: Set the random seed (default: 0)
- `--timeout`, `-t`: Set the timeout for waiting for environment connection (default: 60 seconds)
- `--episodes`, `-e`: Set the number of episodes to run (default: 5)
- `--max-steps`, `-ms`: Set the maximum number of steps per episode (default: 1000)
- `--fps`, `-f`: Set the rendering frame rate (default: 30)
- `--num-agents`, `-na`: Set the number of agents to use (up to 4, default: 1)
- `--agent1`, `-a1`: Path to a Python file containing an Agent class for agent 1
- `--agent2`, `-a2`: Path to a Python file containing an Agent class for agent 2
- `--agent3`, `-a3`: Path to a Python file containing an Agent class for agent 3
- `--agent4`, `-a4`: Path to a Python file containing an Agent class for agent 4

Examples:

```bash
# Connect to an already running Unity editor
python -m mlgame3d

# Launch a Unity game and run 10 episodes
python -m mlgame3d --episodes 10 path/to/your/game.exe
# Or using short options
python -m mlgame3d -e 10 path/to/your/game.exe

# Run without graphics
python -m mlgame3d --no-graphics path/to/your/game.exe
# Or using short options
python -m mlgame3d -ng path/to/your/game.exe

# Set random seed and worker ID
python -m mlgame3d -s 42 -w 1 path/to/your/game.exe

# Use 2 agents, with a custom agent for the first one
python -m mlgame3d --num-agents 2 --agent1 examples/custom_agent.py path/to/your/game.exe
# Or using short options
python -m mlgame3d -na 2 -a1 examples/custom_agent.py path/to/your/game.exe

# Use 4 agents, with custom agents for all of them
python -m mlgame3d -na 4 -a1 agent1.py -a2 agent2.py -a3 agent3.py -a4 agent4.py path/to/your/game.exe
```

### Code Interface

You can also use this framework in your Python code:

```python
from mlgame3d.game_env import GameEnvironment
from mlgame3d.agent import RandomAgent
from mlgame3d.game_runner import GameRunner

# Create the environment with multiple agents
env = GameEnvironment(
    file_name="YourUnityGame.exe",  # Or None to connect to a running Unity editor
    worker_id=0,
    no_graphics=False,
    num_agents=2  # Specify the number of agents (up to 4)
)

# Get information about the action space
action_space_info = env.get_action_space_info()

# Create multiple agents
agent1 = RandomAgent(action_space_info, name="Agent1")
agent2 = RandomAgent(action_space_info, name="Agent2")

# Create a game runner with multiple agents
runner = GameRunner(
    env=env,
    agents=[agent1, agent2],  # Pass a list of agents
    max_episodes=5,
    max_steps_per_episode=1000,
    render=True,
    render_fps=30
)

# Run the game
stats = runner.run()

# Print statistics
print(f"Mean Total Reward: {stats['mean_reward']:.2f}")
print(f"Agent 1 Mean Reward: {stats['agent_mean_rewards'][0]:.2f}")
print(f"Agent 2 Mean Reward: {stats['agent_mean_rewards'][1]:.2f}")

# Close the environment
env.close()
```

### Creating Custom Agents

You can create custom agents in two ways:

1. By inheriting from the `Agent` class (traditional approach)
2. By creating a standalone `MLPlay` class (simplified approach)

#### Option 1: Using the Agent Class

You can create custom agents by inheriting from the `Agent` class. These can be defined in your code or in external Python files that can be loaded at runtime.

Here's an example of a custom agent:

```python
import numpy as np
from mlgame3d.agent import Agent

class MyAgent(Agent):
    def __init__(self, action_space_info, name="MyAgent"):
        super().__init__(name)
        self.action_space_info = action_space_info
    
    def update(self, observations, reward=0.0, done=False, info=None):
        # Update agent state
        super().update(observations, reward, done, info)
        
        # Implement your decision logic
        # This is just a simple example that returns all zeros
        if self.action_space_info.is_continuous():
            return np.zeros(self.action_space_info.continuous_size)
        elif self.action_space_info.is_discrete():
            return np.zeros(len(self.action_space_info.discrete_branches), dtype=np.int32)
        else:
            continuous = np.zeros(self.action_space_info.continuous_size)
            discrete = np.zeros(len(self.action_space_info.discrete_branches), dtype=np.int32)
            return (continuous, discrete)
```

#### Option 2: Using the MLPlay Class (Recommended)

You can create a standalone `MLPlay` class without inheriting from any base class. This approach is simpler and more flexible.

Requirements for `MLPlay` class:
1. The class must be named `MLPlay`
2. The class must implement `__init__`, `update`, and `reset` methods

Here's an example of a minimal `MLPlay` class:

```python
import numpy as np
from typing import Dict, Any

class MLPlay:
    def __init__(self, action_space_info=None):
        # Initialize your agent
        pass
        
    def reset(self):
        # Reset your agent for a new episode
        pass
        
    def update(self, observations, reward=0.0, done=False, info=None):
        # Process observations and choose an action
        # This is a simple example that returns a random 2D movement vector
        action = np.random.uniform(-1, 1, 2)
        
        # Normalize the action vector
        if np.linalg.norm(action) > 0:
            action = action / np.linalg.norm(action)
            
        return action
```

### Loading External Agent Files

You can create custom agents in separate Python files and load them at runtime using the command-line interface. The framework will automatically find and instantiate either an `MLPlay` class or an `Agent` subclass in the file.

Requirements for external agent files:
1. The file must contain either:
   - A class named `MLPlay` with `__init__`, `update`, and `reset` methods, or
   - A class that inherits from `Agent`
2. If using the `Agent` approach, the class must have a constructor that accepts at least `action_space_info` as its first parameter

Example of an external agent file using `MLPlay` (`simple_mlplay.py`):

```python
import numpy as np
from typing import Dict, Any

class MLPlay:
    def __init__(self, action_space_info=None):
        self.step_counter = 0
        
    def reset(self):
        self.step_counter = 0
        
    def update(self, observations, reward=0.0, done=False, info=None):
        self.step_counter += 1
        
        # Alternate between different actions
        if self.step_counter % 2 == 0:
            return np.array([1.0, 0.0])  # Move right
        else:
            return np.array([0.0, 1.0])  # Move forward
```

Example of an external agent file using `Agent` (`custom_agent.py`):

```python
import numpy as np
from mlgame3d.agent import Agent

class CustomAgent(Agent):
    def __init__(self, action_space_info, name="CustomAgent"):
        super().__init__(name)
        self.action_space_info = action_space_info
        
    def update(self, observations, reward=0.0, done=False, info=None):
        # Update agent state
        super().update(observations, reward, done, info)
        
        # Implement your decision logic here
        # This is a simple example that alternates between different actions
        if self.step_count % 2 == 0:
            return np.array([1.0, 0.0])  # Move right
        else:
            return np.array([0.0, 1.0])  # Move forward
```

You can then use these agents with the command-line interface:

```bash
# Using MLPlay class
python -m mlgame3d --agent1 simple_mlplay.py path/to/your/game.exe

# Using Agent class
python -m mlgame3d --agent1 custom_agent.py path/to/your/game.exe
```

Or load them programmatically:

```python
from mlgame3d.agent_loader import create_agent_from_file

# Create an agent from an external file (works with both MLPlay and Agent)
agent = create_agent_from_file("simple_mlplay.py", action_space_info)
```

## Framework Structure

- `game_env.py`: Provides the `GameEnvironment` class for communicating with Unity games
- `agent.py`: Provides the `Agent` base class and several predefined agent classes
- `game_runner.py`: Provides the `GameRunner` class for running games and collecting statistics
- `agent_loader.py`: Provides functionality for loading agent classes from external Python files
- `__main__.py`: Provides the command-line interface
- `examples/`: Contains example agent implementations

## Notes

- Make sure your Unity game has integrated the ML-Agents package
- If you want to connect to a Unity editor, make sure the editor is running and the game scene is loaded
- If you want to connect to a Unity executable, make sure to provide the correct file path
- When using multiple agents, make sure your Unity environment supports the requested number of agents

## Contributing

Pull requests and issues are welcome to improve this framework.

## License

MIT
