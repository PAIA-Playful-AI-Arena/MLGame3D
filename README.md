# MLGame 3D

A framework for playing Unity games with Python agents using ML-Agents for communication with Unity.

## Features

- Simplified communication with Unity games
- Basic agent classes including random agents, keyboard-controlled agents, and human agents
- Game runner for running games and collecting statistics
- Easy to extend and customize
- Command-line interface support

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

# Install in development mode
pip install -e .
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
```

### Code Interface

You can also use this framework in your Python code:

```python
from mlgame3d.game_env import GameEnvironment
from mlgame3d.agent import RandomAgent
from mlgame3d.game_runner import GameRunner

# Create the environment
env = GameEnvironment(
    file_name="YourUnityGame.exe",  # Or None to connect to a running Unity editor
    worker_id=0,
    no_graphics=False
)

# Get information about the action space
action_space_info = env.get_action_space_info()

# Create an agent
agent = RandomAgent(action_space_info)

# Create a game runner
runner = GameRunner(
    env=env,
    agent=agent,
    max_episodes=5,
    max_steps_per_episode=1000,
    render=True,
    render_fps=30
)

# Run the game
stats = runner.run()

# Print statistics
print(f"Mean Reward: {stats['mean_reward']:.2f}")

# Close the environment
env.close()
```

### Creating Custom Agents

You can create custom agents by inheriting from the `Agent` class:

```python
import numpy as np
from mlgame3d.agent import Agent

class MyAgent(Agent):
    def __init__(self, action_space_info, name="MyAgent"):
        super().__init__(name)
        self.action_space_info = action_space_info
    
    def act(self, observations):
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

### Using Keyboard-Controlled Agents

You can create keyboard-controlled agents by inheriting from the `KeyboardAgent` class:

```python
import numpy as np
from mlgame3d.agent import KeyboardAgent

class MyKeyboardAgent(KeyboardAgent):
    def __init__(self, action_space_info, name="MyKeyboardAgent"):
        super().__init__(action_space_info, name)
        
        # Define keyboard mappings
        self.key_action_map = {
            'w': np.array([0, 1]),   # Forward
            's': np.array([0, -1]),  # Backward
            'a': np.array([-1, 0]),  # Turn left
            'd': np.array([1, 0]),   # Turn right
        }
```

## Framework Structure

- `game_env.py`: Provides the `GameEnvironment` class for communicating with Unity games
- `agent.py`: Provides the `Agent` base class and several predefined agent classes
- `game_runner.py`: Provides the `GameRunner` class for running games and collecting statistics
- `__main__.py`: Provides the command-line interface

## Notes

- Make sure your Unity game has integrated the ML-Agents package
- If you want to connect to a Unity editor, make sure the editor is running and the game scene is loaded
- If you want to connect to a Unity executable, make sure to provide the correct file path

## Contributing

Pull requests and issues are welcome to improve this framework.

## License

MIT
