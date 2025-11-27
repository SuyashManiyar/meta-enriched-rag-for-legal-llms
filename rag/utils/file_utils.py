"""File system utilities"""

import os
import json
from pathlib import Path
from typing import List, Dict


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def list_files(directory: str, extensions: List[str]) -> List[Path]:
    """List all files with given extensions in directory"""
    path = Path(directory)
    files = []
    for ext in extensions:
        files.extend(path.glob(f"*{ext}"))
    return sorted(files)


def save_json(data: Dict, filepath: str):
    """Save data as JSON"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
