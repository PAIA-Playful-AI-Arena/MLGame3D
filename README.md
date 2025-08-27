# MLGame3D

A framework for playing Unity games with Python agents using ML-Agents Environment.

## Requirements

- Python 3.11 or higher
- Unity game with ML-Agents package integrated

## Installation

```bash
pip install mlgame3d
```

### Dependencies

This framework depends on:
- `mlgame3d-envs`: Core ML-Agents environment wrapper
- `pandas`: For data processing and CSV output functionality

## Usage

### Command-line Interface

MLGame3D provides a command-line interface that allows you to launch games directly from the command line:

```bash
python -m mlgame3d [options] <Unity game executable>
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
- `--fps`, `-f`: Target number of frames per second for rendering (default: 60)
- `--time-scale`, `-ts`: Time scale factor for the simulation. Higher values make the simulation run faster. Note: Values less than 1.0 will be clamped to 1.0 by Unity and have no effect (default: 1.0)
- `--decision-period`, `-dp`: Number of FixedUpdate steps between AI decisions. Should be a multiple of 20ms. Range: 1-20 (default: 5)
- `--ai`, `-i`: Control mode for an instance (auto-numbered). Can be specified multiple times. Each occurrence is equivalent to `--ai1`, `--ai2`, etc. in order.
- `--ai1`, `-i1`: Control mode for instance 1. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default).
- `--ai2`, `-i2`: Control mode for instance 2. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default).
- `--ai3`, `-i3`: Control mode for instance 3. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default).
- `--ai4`, `-i4`: Control mode for instance 4. Can be a path to a Python file containing an MLPlay class, 'hidden', or 'manual' (default).
- `--game-param`, `-gp`: Game parameter in the format KEY VALUE. Can be specified multiple times for different parameters.
- `--result-output-file`, `-o`: Path to a CSV file where result data will be saved. Each episode's result will be appended to this file with the episode column always appearing first. If the file path doesn't end with '.csv', it will be automatically added.

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

# Use 2 AI instances, with a custom MLPlay for the first one and manual control for the second one
python -m mlgame3d -i examples/simple_mlplay.py -i manual path/to/your/game.exe

# Use 3 AI instances: first controlled by MLPlay, second hidden, third controlled by another MLPlay
python -m mlgame3d -i mlplay1.py -i hidden -i mlplay2.py path/to/your/game.exe

# Specify control modes for specific player positions
python -m mlgame3d -i1 mlplay1.py -i2 manual -i3 mlplay2.py -i4 hidden path/to/your/game.exe

# Pass game parameters (checkpoint_count and checkpoint_mode)
python -m mlgame3d -i examples/simple_mlplay.py -gp checkpoint_count 10 -gp checkpoint_mode random path/to/your/game.exe

# Set simulation parameters (fps, time-scale, and decision-period)
python -m mlgame3d -f 60 -ts 2.0 -dp 10 path/to/your/game.exe

# Save result data to a CSV file
python -m mlgame3d -i examples/simple_mlplay.py -o results.csv path/to/your/game.exe
# Or specify a path without .csv extension (it will be added automatically)
python -m mlgame3d -i examples/simple_mlplay.py -o results/game_results path/to/your/game.exe
```

### Code Interface

You can also use this framework in your Python code:

```python
from mlgame3d.game_env import GameEnvironment
from mlgame3d.mlplay import RandomMLPlay
from mlgame3d.game_runner import GameRunner

# Create the environment with controlled players
env = GameEnvironment(
    file_name="YourUnityGame.exe",  # Or None to connect to a running Unity editor
    worker_id=0,
    no_graphics=False,
    fps=60,  # Target frames per second for rendering
    time_scale=1.0,  # Time scale factor for simulation speed
    decision_period=5,  # Number of FixedUpdate steps between AI decisions
    controlled_players=[0, 1],  # Control P1 and P2
    control_modes=["mlplay", "mlplay"],  # Both controlled by MLPlay
    game_parameters=[("checkpoint_count", 10), ("checkpoint_mode", "random")],  # Game parameters
    result_output_file="results.csv"  # Save result data to this CSV file
)

# Get information about the action space for each behavior
action_space_info_p1 = env.get_action_space_info(env.behavior_names[0])
action_space_info_p2 = env.get_action_space_info(env.behavior_names[1])

# Create MLPlay instances
mlplay1 = RandomMLPlay(action_space_info_p1, name="MLPlay1")
mlplay2 = RandomMLPlay(action_space_info_p2, name="MLPlay2")

# Create a mapping from MLPlay index to behavior name
mlplay_to_behavior_map = {
    0: env.behavior_names[0],  # First MLPlay controls first behavior
    1: env.behavior_names[1]   # Second MLPlay controls second behavior
}

# Create a game runner
runner = GameRunner(
    env=env,
    mlplays=[mlplay1, mlplay2],
    max_episodes=5,
    render=True,
    mlplay_timeout=0.1,  # Timeout for MLPlay actions in seconds
    game_parameters={"checkpoint_count": 10, "checkpoint_mode": "random"},
    mlplay_to_behavior_map=mlplay_to_behavior_map
)

# Run the game
runner.run()

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
    def __init__(self, observation_structure=None, action_space_info=None, name=None, game_params=None):
        """
        Initialize your MLPlay instance.
        
        Args:
            observation_structure: Dictionary describing observation space structure
            action_space_info: ActionSpec object describing action space
            name: Name for this MLPlay instance
            game_params: Dictionary of game parameters
        """
        self.action_space_info = action_space_info
        self.name = name or "MLPlay"
        self.observation_structure = observation_structure
        self.game_params = game_params or {}
        
    def reset(self):
        """Reset your MLPlay instance for a new episode."""
        pass
        
    def update(self, observations, done=False, info=None):
        """
        Process observations and choose an action.
        
        Args:
            observations: Dictionary of observations from the environment
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            Action in the format required by the action space
        """
        # Determine action format based on action space
        if self.action_space_info.is_continuous():
            # Pure continuous action space
            return np.random.uniform(-1, 1, self.action_space_info.continuous_size)
            
        elif self.action_space_info.is_discrete():
            # Pure discrete action space
            return np.array([
                np.random.randint(0, branch) 
                for branch in self.action_space_info.discrete_branches
            ], dtype=np.int32)
            
        else:
            # Hybrid action space (both continuous and discrete)
            continuous = np.random.uniform(-1, 1, self.action_space_info.continuous_size)
            discrete = np.array([
                np.random.randint(0, branch) 
                for branch in self.action_space_info.discrete_branches
            ], dtype=np.int32)
            return (continuous, discrete)
```

### Loading External MLPlay Files

You can create custom MLPlay classes in separate Python files and load them at runtime using the command-line interface. The framework will automatically find and instantiate the `MLPlay` class in the file.

Requirements for external MLPlay files:
1. The file must contain a class named `MLPlay`
2. The class must implement `__init__`, `update`, and `reset` methods
3. The `__init__` method should accept the parameters: `observation_structure`, `action_space_info`, `name`, and `game_params`

Example of an external MLPlay file (`simple_mlplay.py`):

```python
import numpy as np
from typing import Dict, Any

class MLPlay:
    def __init__(self, observation_structure=None, action_space_info=None, name=None, game_params=None):
        """
        Initialize the MLPlay instance.
        
        Args:
            observation_structure: Dictionary describing observation space structure
            action_space_info: ActionSpec object describing action space
            name: Name for this MLPlay instance
            game_params: Dictionary of game parameters
        """
        self.action_space_info = action_space_info
        self.name = name or "SimpleMLPlay"
        self.observation_structure = observation_structure
        self.game_params = game_params or {}
        self.step_counter = 0
        
    def reset(self):
        """Reset for a new episode."""
        self.step_counter = 0
        
    def update(self, observations, done=False, info=None):
        """
        Process observations and choose an action.
        
        Args:
            observations: Dictionary of observations from the environment
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            Action in the format required by the action space
        """
        self.step_counter += 1
        
        # Simple strategy: alternate between different movement patterns
        if self.action_space_info.is_continuous():
            # Pure continuous action space
            if self.step_counter % 4 < 2:
                return np.array([1.0, 0.0])  # Move right
            else:
                return np.array([0.0, 1.0])  # Move forward
                
        elif self.action_space_info.is_discrete():
            # Pure discrete action space
            return np.array([self.step_counter % 2], dtype=np.int32)
            
        else:
            # Hybrid action space
            if self.step_counter % 4 < 2:
                continuous = np.array([1.0, 0.0])  # Move right
            else:
                continuous = np.array([0.0, 1.0])  # Move forward
                
            discrete = np.array([self.step_counter % 2], dtype=np.int32)
            return (continuous, discrete)
```

You can then use these MLPlay classes with the command-line interface:

```bash
python -m mlgame3d -i simple_mlplay.py path/to/your/game.exe
```

Or specify a specific player position:

```bash
python -m mlgame3d -i1 simple_mlplay.py path/to/your/game.exe
```

Or load them programmatically:

```python
from mlgame3d.mlplay_loader import create_mlplay_from_file

# You need to get these from the environment first
env = GameEnvironment(file_name="game.exe", controlled_players=[0], control_modes=["mlplay"])
behavior_name = env.behavior_names[0]
observation_structure = env.get_observation_structure(behavior_name)
action_space_info = env.get_action_space_info(behavior_name)

# Create an MLPlay instance from an external file
mlplay = create_mlplay_from_file(
    file_path="simple_mlplay.py", 
    observation_structure=observation_structure,
    action_space_info=action_space_info,
    name="MyCustomMLPlay",
    game_parameters={"some_param": "value"}
)
```

## Framework Structure

### Core Components

- `game_env.py`: Provides the `GameEnvironment` class for communicating with Unity games via ML-Agents
- `mlplay.py`: Provides the `RandomMLPlay` class for generating random actions
- `game_runner.py`: Provides the `GameRunner` class for running games and collecting statistics
- `mlplay_loader.py`: Provides functionality for loading MLPlay classes from external Python files
- `__main__.py`: Provides the command-line interface

### Side Channels

The framework includes several side channels for enhanced Unity communication:

- `observation_structure_side_channel.py`: Handles observation space structure information
- `player_control_side_channel.py`: Manages player control modes (manual, MLPlay, hidden)
- `game_parameters_side_channel.py`: Sends game parameters to Unity
- `ranking_side_channel.py`: Receives ranking/result data from Unity
- `keyboard_state_side_channel.py`: Manages keyboard input state for manual control

### Examples

- `examples/simple_mlplay.py`: Basic MLPlay implementation with random movement
- `examples/proly_mlplay.py`: Game-specific MLPlay example for Proly racing game

## Notes

- Make sure your Unity game has integrated the ML-Agents package
- If you want to connect to a Unity editor, make sure the editor is running and the game scene is loaded
- If you want to connect to a Unity executable, make sure to provide the correct file path
- When using multiple MLPlay instances, make sure your Unity environment supports the requested number of agents
- Control modes:
  - `manual`: Player is controlled manually via keyboard/gamepad
  - `hidden`: Player is hidden (not visible in the game)
  - Python file path: Player is controlled by an MLPlay instance loaded from the specified file
- Result data:
  - When using the `--result-output-file` option, result data will be saved to the specified CSV file
  - Each episode's result data will be appended to the file with the episode column always appearing first
  - If the file path doesn't end with '.csv', it will be automatically added
  - CSV output includes episode number, player rankings, and any additional game-specific metrics
- Action Space Support:
  - Supports continuous action spaces (floating-point values)
  - Supports discrete action spaces (integer selections)
  - Supports hybrid action spaces (combination of continuous and discrete)
- Side Channel Communication:
  - Real-time parameter passing to Unity during runtime
  - Observation structure discovery for dynamic adaptation
  - Player control mode switching without restarting the environment

## Contributing

Pull requests and issues are welcome to improve this framework. Please visit the [GitHub repository](https://github.com/PAIA-Playful-AI-Arena/MLGame3D) for more information.

## Project Links

- **Homepage**: https://github.com/PAIA-Playful-AI-Arena/MLGame3D
- **Repository**: https://github.com/PAIA-Playful-AI-Arena/MLGame3D
- **Documentation**: See this README for detailed usage instructions
- **Issues**: Report bugs and request features on GitHub

## License

MIT License - see LICENSE file for details.
