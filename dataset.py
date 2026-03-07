"""
Dataset loading for APO optimization.
Data format: list of dicts, each must have "input" and "output" keys.
Agent 只处理每条数据的 input，算法用 output 做评估。
"""
import json
from typing import Any, List, Dict


def load_dataset_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载数据集。每条数据必须包含 "input" 和 "output" 两个 key。

    Expected JSON format:
    [
        {"input": ..., "output": ...},
        {"input": ..., "output": ...},
        ...
    ]

    Args:
        file_path: JSON 文件路径

    Returns:
        List of dicts with "input" and "output" keys
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {file_path}: {e}")
        return []

    if not isinstance(data, list):
        print(f"Error: JSON root must be an array, got {type(data)}")
        return []

    result = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: Skip item {i}, not a dict")
            continue
        if "input" not in item or "output" not in item:
            print(f"Warning: Skip item {i}, missing 'input' or 'output' key")
            continue
        result.append(item)
    return result
