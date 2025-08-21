"""
MLPlay Loader Module

This module provides functionality for loading MLPlay classes from external Python files.
"""

import os
import sys
import importlib.util
import inspect
import traceback
from typing import Type, Optional, Any

def load_mlplay_class(file_path: str) -> Optional[Type[Any]]:
    """
    Load an MLPlay class from an external Python file.
    
    Args:
        file_path: Path to the Python file containing the class.
        
    Returns:
        The MLPlay class if found, None otherwise.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MLPlay file not found: {file_path}")
        
    if not file_path.endswith('.py'):
        raise ValueError(f"MLPlay file must be a Python file: {file_path}")
        
    # Get the module name from the file path
    module_name = os.path.basename(file_path).replace('.py', '')
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load module from {file_path}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Try to find an MLPlay class
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and name == "MLPlay":
            return obj
    
    raise ValueError(f"No MLPlay class found in {file_path}")

def create_mlplay_from_file(file_path: str, observation_structure: dict, action_space_info, name: Optional[str] = None, game_parameters: Optional[dict] = None) -> Any:
    """
    Create an MLPlay instance from an external Python file.
    
    Args:
        file_path: Path to the Python file containing the MLPlay class.
        action_space_info: Information about the action space to pass to the constructor.
        name: Optional name for the MLPlay instance. If None, the class name will be used.
        game_parameters: Optional game parameters to pass to the constructor.

    Returns:
        An instance of the MLPlay class.
    """
    cls = load_mlplay_class(file_path)
    
    # Create an MLPlay instance
    try:
        ml_play_instance = cls(observation_structure, action_space_info, name, game_params=game_parameters)
    except Exception as e:
        raise ValueError(f"Error creating MLPlay instance: {e}")
    
    # Set the name attribute if provided
    if name and not hasattr(ml_play_instance, 'name'):
        ml_play_instance.name = name
    elif not hasattr(ml_play_instance, 'name'):
        ml_play_instance.name = f"MLPlay_{os.path.basename(file_path).replace('.py', '')}"
    
    return ml_play_instance

def validate_mlplay_file(file_path: str) -> bool:
    """
    Validate that an external Python file contains a valid MLPlay class.
    
    Args:
        file_path: Path to the Python file to validate.
        
    Returns:
        True if the file contains a valid MLPlay class, False otherwise.
    """
    try:
        cls = load_mlplay_class(file_path)
        
        # Check if it has the required methods
        has_init = hasattr(cls, '__init__') and callable(getattr(cls, '__init__'))
        has_update = hasattr(cls, 'update') and callable(getattr(cls, 'update'))
        has_reset = hasattr(cls, 'reset') and callable(getattr(cls, 'reset'))
        
        if not (has_init and has_update):
            print(f"Warning: MLPlay class in {file_path} does not have the required methods (__init__, update).")
            return False
        
        if not has_reset:
            print(f"Warning: MLPlay class in {file_path} does not have a reset method.")
        
        return True
    except Exception as e:
        print(f"Error validating MLPlay file {file_path}: {e}")
        traceback.print_exc()
        return False
