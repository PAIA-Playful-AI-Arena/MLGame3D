# Playing Proly with MLGame3D

This guide explains how to use the MLGame3D framework to play the Proly game with Python agents.

## Overview

Proly is a racing game where players navigate through checkpoints while avoiding obstacles and other players. The MLGame3D framework allows you to create Python agents that can play this game.

## Setup

1. Make sure you have both the MLGame3D framework and the Proly game installed:
   - MLGame3D should be installed in development mode: `pip install -e .`
   - Proly should be built and ready to run

2. Ensure that the Proly game has ML-Agents integrated correctly.

## Running Proly with MLGame3D

You can run Proly with MLGame3D using the following command:

```bash
python -m mlgame3d [options] path/to/Proly-build/Proly.exe
```

### Options

- `--num-agents`, `-na`: Set the number of agents (1-4, default: 1)
- `--agent1`, `-a1`: Path to a Python file for agent 1 (default: RandomAgent)
- `--agent2`, `-a2`: Path to a Python file for agent 2 (default: RandomAgent)
- `--agent3`, `-a3`: Path to a Python file for agent 3 (default: RandomAgent)
- `--agent4`, `-a4`: Path to a Python file for agent 4 (default: RandomAgent)
- `--episodes`, `-e`: Number of episodes to run (default: 5)
- `--max-steps`, `-ms`: Maximum steps per episode (default: 1000)
- `--no-graphics`, `-ng`: Run without graphics (useful for training)
- `--seed`, `-s`: Random seed for reproducibility

For a full list of options, run:

```bash
python -m mlgame3d --help
```

### Examples

1. Run Proly with a single random agent:

```bash
python -m mlgame3d path/to/Proly-build/Proly.exe
```

2. Run Proly with the provided ProlyAgent:

```bash
python -m mlgame3d --agent1 examples/proly_agent.py path/to/Proly-build/Proly.exe
```

3. Run Proly with 2 agents, one custom and one random:

```bash
python -m mlgame3d --num-agents 2 --agent1 examples/proly_agent.py path/to/Proly-build/Proly.exe
```

4. Run Proly with 4 different agents:

```bash
python -m mlgame3d --num-agents 4 --agent1 agent1.py --agent2 agent2.py --agent3 agent3.py --agent4 agent4.py path/to/Proly-build/Proly.exe
```

## Observations

The Proly game provides the following observations to agents:

1. Target position (x, y, z): Position of the next checkpoint
2. Current position (x, y, z): Position of the agent
3. Current velocity (x, z): Velocity of the agent
4. Health information: Current health and normalized health (if available)
5. Checkpoint information: Index of the last checkpoint passed
6. Time information: Current time in the game
7. Other players information: Relative positions and velocities of other players

## Actions

Agents can control their movement with a 2D continuous action vector:

- `[0]`: X-axis movement (-1 to 1, where -1 is left, 1 is right)
- `[1]`: Z-axis movement (-1 to 1, where -1 is backward, 1 is forward)

## Rewards

The Proly game provides the following rewards:

- +1.0 for passing a checkpoint
- -1.0 for falling off the platform
- -0.1 for colliding with obstacles or other players

## Creating Custom Agents

You can create custom agents by inheriting from the `Agent` class in the MLGame3D framework. See `examples/proly_agent.py` for an example.

The key methods to implement are:

- `__init__(self, action_space_info, name)`: Initialize the agent
- `reset(self)`: Reset the agent for a new episode
- `act(self, observations)`: Choose an action based on observations
- `observe(self, observations, reward, done, info)`: Process the results of an action

## Tips for Creating Effective Agents

1. **Understand the observations**: Parse the observations correctly to extract useful information.
2. **Navigate to checkpoints**: Focus on reaching the next checkpoint.
3. **Avoid obstacles and other players**: Implement collision avoidance.
4. **Manage health**: Be careful not to lose health by falling or colliding.
5. **Optimize for time**: Try to complete the course as quickly as possible.

## Debugging

If your agent is not behaving as expected, you can add print statements to debug:

- Print observations to understand what the agent is seeing
- Print actions to understand what the agent is doing
- Print rewards to understand how the agent is being rewarded

The ProlyAgent example includes a `_print_debug_info` method that demonstrates how to print useful debugging information.
