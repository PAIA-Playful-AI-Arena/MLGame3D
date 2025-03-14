"""
Agent Loader Module

This module provides functionality for loading agent classes from external Python files.
"""

import os
import sys
import importlib.util
import inspect
from typing import Type, List, Optional

from mlgame3d.agent import Agent

def load_agent_class(file_path: str) -> Optional[Type[Agent]]:
    """
    Load an Agent class from an external Python file.
    
    Args:
        file_path: Path to the Python file containing the Agent class.
        
    Returns:
        The Agent class if found, None otherwise.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Agent file not found: {file_path}")
        
    if not file_path.endswith('.py'):
        raise ValueError(f"Agent file must be a Python file: {file_path}")
        
    # Get the module name from the file path
    module_name = os.path.basename(file_path).replace('.py', '')
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load module from {file_path}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Find the Agent class in the module
    agent_classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Agent) and obj != Agent:
            agent_classes.append(obj)
            
    if not agent_classes:
        raise ValueError(f"No Agent subclass found in {file_path}")
        
    if len(agent_classes) > 1:
        print(f"Warning: Multiple Agent subclasses found in {file_path}. Using the first one: {agent_classes[0].__name__}")
        
    return agent_classes[0]

def create_agent_from_file(file_path: str, action_space_info, name: Optional[str] = None) -> Agent:
    """
    Create an Agent instance from an external Python file.
    
    Args:
        file_path: Path to the Python file containing the Agent class.
        action_space_info: Information about the action space to pass to the Agent constructor.
        name: Optional name for the agent. If None, the class name will be used.
        
    Returns:
        An instance of the Agent class.
    """
    agent_class = load_agent_class(file_path)
    
    # Check if the agent class constructor accepts the action_space_info parameter
    signature = inspect.signature(agent_class.__init__)
    parameters = list(signature.parameters.keys())
    
    # The first parameter is 'self', so we check the second one
    if len(parameters) > 1 and parameters[1] == 'action_space_info':
        if name:
            # Check if the constructor accepts a name parameter
            if 'name' in parameters:
                return agent_class(action_space_info, name=name)
            else:
                return agent_class(action_space_info)
        else:
            return agent_class(action_space_info)
    else:
        raise ValueError(f"Agent class {agent_class.__name__} does not have the expected constructor signature. "
                         f"Expected: __init__(self, action_space_info, name=None)")

def validate_agent_file(file_path: str) -> bool:
    """
    Validate that an external Python file contains a valid Agent subclass.
    
    Args:
        file_path: Path to the Python file to validate.
        
    Returns:
        True if the file contains a valid Agent subclass, False otherwise.
    """
    try:
        agent_class = load_agent_class(file_path)
        return True
    except Exception as e:
        print(f"Error validating agent file {file_path}: {e}")
        return False
