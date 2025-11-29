"""
Visualization utilities for multimodal emotion recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import itertools

from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(
    history: Union[Dict, 'tf.keras.callbacks.History'],
    metrics: List[str] = ['loss', 'accuracy'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Training history (dict or History object)
        metrics: List of metrics to plot
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert History object to dict if needed
    if hasattr(history, 'history'):
        history = history.history
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
        
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize values
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_emotion_distribution(
    labels: np.ndarray,
    class_names: List[str],
    title: str = 'Emotion Distribution',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot distribution of emotions in dataset.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        colors: List of colors for bars
        
    Returns:
        Matplotlib figure
    """
    # Count occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Map to class names
    label_counts = {class_names[i]: count for i, count in zip(unique, counts)}
    
    # Default colors
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = range(len(class_names))
    bars = ax.bar(x_pos, [label_counts.get(name, 0) for name in class_names], color=colors)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add count labels on bars
    for bar, count in zip(bars, [label_counts.get(name, 0) for name in class_names]):
        height = bar.get_height()
        ax.annotate(
            f'{count}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Emotion distribution plot saved to {save_path}")
    
    return fig


def visualize_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    class_names: List[str],
    num_samples: int = 16,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 16)
) -> plt.Figure:
    """
    Visualize model predictions on sample images.
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        pred_probs: Prediction probabilities
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = images[i]
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img.squeeze()
        
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        
        # Get labels
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]
        confidence = pred_probs[i][pred_labels[i]]
        
        # Color based on correctness
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        ax.set_title(
            f'True: {true_label}\nPred: {pred_label} ({confidence:.1%})',
            color=color,
            fontsize=10
        )
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    n_classes = len(class_names)
    
    # Binarize labels if needed
    if len(y_true.shape) == 1:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
    else:
        y_true_bin = y_true
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f'{class_name} (AUC = {roc_auc:.2f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    return fig


def plot_learning_rate_schedule(
    schedule_fn,
    epochs: int,
    steps_per_epoch: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Plot learning rate schedule.
    
    Args:
        schedule_fn: Learning rate schedule function
        epochs: Number of epochs
        steps_per_epoch: Steps per epoch
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    total_steps = epochs * steps_per_epoch
    steps = np.arange(total_steps)
    lrs = [schedule_fn(step) for step in steps]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(steps / steps_per_epoch, lrs, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate schedule saved to {save_path}")
    
    return fig


def create_emotion_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Create detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report
        
    Returns:
        Classification report string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    # Add header
    full_report = "=" * 60 + "\n"
    full_report += "EMOTION RECOGNITION - CLASSIFICATION REPORT\n"
    full_report += "=" * 60 + "\n\n"
    full_report += report
    
    # Add overall accuracy
    accuracy = np.mean(y_true == y_pred)
    full_report += f"\nOverall Accuracy: {accuracy:.4f} ({accuracy:.2%})\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(full_report)
        print(f"Classification report saved to {save_path}")
    
    return full_report


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to display
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importances[indices], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def visualize_audio_features(
    audio_features: np.ndarray,
    feature_type: str = 'mfcc',
    sr: int = 22050,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Visualize audio features (MFCC, mel spectrogram, etc.).
    
    Args:
        audio_features: Audio feature array
        feature_type: Type of features ('mfcc', 'melspec')
        sr: Sample rate
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    img = ax.imshow(
        audio_features,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    
    ax.set_xlabel('Time Frame', fontsize=12)
    ax.set_ylabel(f'{feature_type.upper()} Coefficient', fontsize=12)
    ax.set_title(f'{feature_type.upper()} Features', fontsize=14)
    
    plt.colorbar(img, ax=ax, label='Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Audio features visualization saved to {save_path}")
    
    return fig


def create_comparison_plot(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create comparison plot for multiple models.
    
    Args:
        results: Dictionary of model results {model_name: {metric: value}}
        metric: Metric to compare
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    model_names = list(results.keys())
    values = [results[name].get(metric, 0) for name in model_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, values, color=colors)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14)
    ax.set_ylim([0, max(values) * 1.1])
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=11
        )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig
