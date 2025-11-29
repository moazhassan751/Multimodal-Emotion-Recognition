"""
Helper utilities for multimodal emotion recognition project.
"""

import os
import yaml
import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    ensure_dir(config_path.parent)
    
    with open(config_path, 'w') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_name: str = 'emotion_recognition'
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        ensure_dir(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f'{log_name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Additional TensorFlow settings for reproducibility
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'tensorflow_version': tf.__version__,
        'num_cpus': os.cpu_count(),
        'gpus': [],
        'gpu_available': False
    }
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        info['gpu_available'] = True
        for gpu in gpus:
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                info['gpus'].append({
                    'name': gpu.name,
                    'memory': memory_info
                })
            except:
                info['gpus'].append({'name': gpu.name})
    
    return info


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model_with_metadata(
    model: tf.keras.Model,
    save_path: str,
    metadata: Dict[str, Any],
    save_format: str = 'keras'
) -> None:
    """
    Save model along with metadata.
    
    Args:
        model: Keras model to save
        save_path: Path to save model
        metadata: Metadata dictionary
        save_format: Format to save model ('keras', 'h5', 'savedmodel')
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    
    # Save model
    if save_format == 'keras':
        model.save(save_path.with_suffix('.keras'))
    elif save_format == 'h5':
        model.save(save_path.with_suffix('.h5'))
    elif save_format == 'savedmodel':
        model.save(save_path)
    
    # Save metadata
    metadata['save_timestamp'] = datetime.now().isoformat()
    metadata['model_summary'] = []
    
    # Capture model summary
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    metadata['model_summary'] = string_list
    
    metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def get_class_weights(labels: np.ndarray, num_classes: Optional[int] = None) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class indices to weights
    """
    if num_classes is None:
        num_classes = len(np.unique(labels))
    
    # Count samples per class
    counts = np.bincount(labels, minlength=num_classes)
    
    # Calculate weights (inverse frequency)
    total = len(labels)
    weights = total / (num_classes * counts + 1e-8)
    
    return {i: w for i, w in enumerate(weights)}


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"


def create_experiment_name(
    model_type: str,
    dataset: str,
    timestamp: bool = True
) -> str:
    """
    Create a unique experiment name.
    
    Args:
        model_type: Type of model (facial, speech, fusion)
        dataset: Dataset name
        timestamp: Whether to include timestamp
        
    Returns:
        Experiment name string
    """
    name = f"{model_type}_{dataset}"
    
    if timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"{name}_{ts}"
    
    return name


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """
    Count trainable and non-trainable parameters in model.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    
    return {
        'trainable': int(trainable),
        'non_trainable': int(non_trainable),
        'total': int(trainable + non_trainable)
    }


def print_model_info(model: tf.keras.Model, name: str = "Model") -> None:
    """
    Print model information including architecture and parameters.
    
    Args:
        model: Keras model
        name: Model name for display
    """
    params = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"{'='*60}\n")
    
    model.summary()


def moving_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Calculate moving average of data.
    
    Args:
        data: Input data array
        window_size: Size of moving average window
        
    Returns:
        Smoothed data array
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
