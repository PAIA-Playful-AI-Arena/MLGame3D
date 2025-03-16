# MLGame3D

A framework for playing Unity games with Python MLPlay classes using ML-Agents for communication with Unity.

## Features

- Simplified communication with Unity games
- Simple MLPlay class structure with no inheritance required
- Game runner for running games and collecting statistics
- Easy to extend and customize
- Command-line interface support
- Support for multiple MLPlay instances (up to 4) playing simultaneously
- Dynamic loading of external MLPlay files

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
- `--fps`, `-f`: Set the rendering frame rate (default: 30)
- `--num-agents`, `-na`: Set the number of MLPlay instances to use (up to 4, default: 1)
- `--mlplay1`, `-m1`: Path to a Python file containing an MLPlay class for instance 1
- `--mlplay2`, `-m2`: Path to a Python file containing an MLPlay class for instance 2
- `--mlplay3`, `-m3`: Path to a Python file containing an MLPlay class for instance 3
- `--mlplay4`, `-m4`: Path to a Python file containing an MLPlay class for instance 4

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

# Use 2 MLPlay instances, with a custom MLPlay for the first one
python -m mlgame3d --num-agents 2 --mlplay1 examples/simple_mlplay.py path/to/your/game.exe
# Or using short options
python -m mlgame3d -na 2 -m1 examples/simple_mlplay.py path/to/your/game.exe

# Use 4 MLPlay instances, with custom MLPlay for all of them
python -m mlgame3d -na 4 -m1 mlplay1.py -m2 mlplay2.py -m3 mlplay3.py -m4 mlplay4.py path/to/your/game.exe
```

### Code Interface

You can also use this framework in your Python code:

```python
from mlgame3d.game_env import GameEnvironment
from mlgame3d.mlplay import RandomMLPlay
from mlgame3d.game_runner import GameRunner

# Create the environment with multiple MLPlay instances
env = GameEnvironment(
    file_name="YourUnityGame.exe",  # Or None to connect to a running Unity editor
    worker_id=0,
    no_graphics=False,
    num_agents=2  # Specify the number of MLPlay instances (up to 4)
)

# Get information about the action space
action_space_info = env.get_action_space_info()

# Create multiple MLPlay instances
mlplay1 = RandomMLPlay(action_space_info, name="MLPlay1")
mlplay2 = RandomMLPlay(action_space_info, name="MLPlay2")

# Create a game runner with multiple MLPlay instances
runner = GameRunner(
    env=env,
    mlplays=[mlplay1, mlplay2],  # Pass a list of MLPlay instances
    max_episodes=5,
    render=True,
    render_fps=30
)

# Run the game
stats = runner.run()

# Print statistics
print(f"Mean Total Reward: {stats['mean_reward']:.2f}")
print(f"MLPlay 1 Mean Reward: {stats['mlplay_mean_rewards'][0]:.2f}")
print(f"MLPlay 2 Mean Reward: {stats['mlplay_mean_rewards'][1]:.2f}")

# Close the environment
env.close()
```

### Creating Custom MLPlay Classes

You can create a standalone `MLPlay` class without inheriting from any base class. This approach is simple and flexible.

Requirements for `MLPlay` class:
1. The class must be named `MLPlay`
2. The class must implement `__init__`, `update`, and `reset` methods

Here's an example of a minimal `MLPlay` class:

```python
import numpy as np
from typing import Dict, Any

class MLPlay:
    def __init__(self, action_space_info=None):
        # Initialize your MLPlay instance
        pass
        
    def reset(self):
        # Reset your MLPlay instance for a new episode
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

### Loading External MLPlay Files

You can create custom MLPlay classes in separate Python files and load them at runtime using the command-line interface. The framework will automatically find and instantiate the `MLPlay` class in the file.

Requirements for external MLPlay files:
1. The file must contain a class named `MLPlay`
2. The class must implement `__init__`, `update`, and `reset` methods

Example of an external MLPlay file (`simple_mlplay.py`):

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

You can then use these MLPlay classes with the command-line interface:

```bash
python -m mlgame3d --mlplay1 simple_mlplay.py path/to/your/game.exe
```

Or load them programmatically:

```python
from mlgame3d.mlplay_loader import create_mlplay_from_file

# Create an MLPlay instance from an external file
mlplay = create_mlplay_from_file("simple_mlplay.py", action_space_info)
```

## Framework Structure

- `game_env.py`: Provides the `GameEnvironment` class for communicating with Unity games
- `mlplay.py`: Provides the `RandomMLPlay` class for generating random actions
- `game_runner.py`: Provides the `GameRunner` class for running games and collecting statistics
- `mlplay_loader.py`: Provides functionality for loading MLPlay classes from external Python files
- `__main__.py`: Provides the command-line interface
- `examples/`: Contains example MLPlay implementations

## Notes

- Make sure your Unity game has integrated the ML-Agents package
- If you want to connect to a Unity editor, make sure the editor is running and the game scene is loaded
- If you want to connect to a Unity executable, make sure to provide the correct file path
- When using multiple MLPlay instances, make sure your Unity environment supports the requested number of agents

## Contributing

Pull requests and issues are welcome to improve this framework.

## License

MIT
