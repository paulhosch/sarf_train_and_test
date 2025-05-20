#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import pandas as pd
from typing import Dict, Any, Optional

def ensure_dir(directory: str) -> str:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path to create
        
    Returns:
        The directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary as JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict:
    """
    Load JSON file as dictionary
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

class Timer:
    """Simple timer class for measuring execution time"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize timer
        
        Args:
            name: Optional name for the timer
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start the timer when entering context"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop the timer when exiting context"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        prefix = f"{self.name}: " if self.name else ""
        print(f"{prefix}Completed in {format_time(elapsed)}")
    
    def elapsed(self) -> float:
        """
        Get elapsed time
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time 