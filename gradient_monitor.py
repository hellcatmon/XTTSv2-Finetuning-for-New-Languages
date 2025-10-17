#!/usr/bin/env python3
"""
Gradient Monitoring and Visualization for XTTS Training
Tracks gradient statistics, detects issues, and provides visualization tools

Usage:
    from gradient_monitor import GradientMonitor

    monitor = GradientMonitor(log_dir="logs/gradients", log_interval=100)
    monitor.log_gradients(model, optimizer, step)
    monitor.save_statistics()
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class GradientMonitor:
    """
    Monitor and log gradient statistics during training

    Features:
    - Track gradient norms (total, per-layer, per-parameter)
    - Detect gradient explosion/vanishing
    - Log to TensorBoard or JSON
    - Generate visualization plots
    - Provide recommendations
    """

    def __init__(
        self,
        log_dir: str = "logs/gradients",
        log_interval: int = 100,
        enable_tensorboard: bool = True,
        explosion_threshold: float = 100.0,
        vanishing_threshold: float = 1e-4,
        track_per_layer: bool = True,
        save_plots: bool = True,
    ):
        """
        Initialize GradientMonitor

        Args:
            log_dir: Directory for saving logs and plots
            log_interval: Log every N steps
            enable_tensorboard: Enable TensorBoard logging
            explosion_threshold: Threshold for gradient explosion warning
            vanishing_threshold: Threshold for vanishing gradient warning
            track_per_layer: Track per-layer gradient statistics
            save_plots: Save visualization plots
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval
        self.enable_tensorboard = enable_tensorboard
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.track_per_layer = track_per_layer
        self.save_plots = save_plots

        # Statistics storage
        self.gradient_norms = []
        self.layer_norms = defaultdict(list)
        self.max_grads = []
        self.min_grads = []
        self.mean_grads = []
        self.steps = []

        # Issue tracking
        self.explosions = []
        self.vanishings = []

        # TensorBoard writer
        self.writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.enable_tensorboard = False

    def compute_gradient_stats(
        self, model: torch.nn.Module
    ) -> Dict[str, float]:
        """
        Compute gradient statistics for model

        Args:
            model: PyTorch model

        Returns:
            Dictionary of gradient statistics
        """
        total_norm = 0.0
        num_params = 0
        max_grad = 0.0
        min_grad = float('inf')
        grad_sum = 0.0

        layer_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Compute parameter gradient norm
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1

                # Track min/max
                param_max = param.grad.data.abs().max().item()
                param_min = param.grad.data.abs().min().item()
                max_grad = max(max_grad, param_max)
                min_grad = min(min_grad, param_min)

                # Track mean
                grad_sum += param.grad.data.abs().mean().item()

                # Per-layer statistics
                if self.track_per_layer:
                    layer_name = name.split('.')[0]  # Get top-level module name
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = {'norm': 0.0, 'count': 0}
                    layer_stats[layer_name]['norm'] += param_norm ** 2
                    layer_stats[layer_name]['count'] += 1

        total_norm = total_norm ** 0.5
        mean_grad = grad_sum / num_params if num_params > 0 else 0.0

        # Compute layer norms
        for layer_name in layer_stats:
            layer_stats[layer_name]['norm'] = layer_stats[layer_name]['norm'] ** 0.5

        stats = {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'min_grad': min_grad,
            'mean_grad': mean_grad,
            'num_params': num_params,
            'layer_stats': layer_stats,
        }

        return stats

    def detect_issues(
        self, stats: Dict[str, float], step: int
    ) -> List[str]:
        """
        Detect gradient-related issues

        Args:
            stats: Gradient statistics
            step: Current training step

        Returns:
            List of detected issues
        """
        issues = []

        # Check for gradient explosion
        if stats['total_norm'] > self.explosion_threshold:
            issues.append(f"EXPLOSION: Gradient norm {stats['total_norm']:.2f} exceeds threshold {self.explosion_threshold}")
            self.explosions.append(step)

        # Check for vanishing gradients
        if stats['total_norm'] < self.vanishing_threshold:
            issues.append(f"VANISHING: Gradient norm {stats['total_norm']:.6f} below threshold {self.vanishing_threshold}")
            self.vanishings.append(step)

        # Check for NaN or Inf
        if np.isnan(stats['total_norm']) or np.isinf(stats['total_norm']):
            issues.append(f"NaN/Inf: Invalid gradient norm detected")

        return issues

    def get_recommendations(self, stats: Dict[str, float]) -> List[str]:
        """
        Get training recommendations based on gradient statistics

        Args:
            stats: Gradient statistics

        Returns:
            List of recommendations
        """
        recommendations = []

        # High gradient norm
        if stats['total_norm'] > 10.0:
            recommendations.append("Consider reducing learning rate")
            recommendations.append("Enable gradient clipping (e.g., clip_val=1.0)")

        # Very high gradient norm
        if stats['total_norm'] > 50.0:
            recommendations.append("Gradient norm very high - check for unstable training")
            recommendations.append("Try reducing learning rate by 10x")

        # Low gradient norm
        if stats['total_norm'] < 0.01:
            recommendations.append("Gradients may be too small")
            recommendations.append("Consider increasing learning rate")

        # Very low gradient norm
        if stats['total_norm'] < 0.001:
            recommendations.append("Vanishing gradients detected")
            recommendations.append("Check model architecture and initialization")

        return recommendations

    def log_gradients(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: int = 0,
        loss: Optional[float] = None,
    ) -> None:
        """
        Log gradient statistics

        Args:
            model: PyTorch model
            optimizer: Optimizer (optional, for learning rate logging)
            step: Current training step
            loss: Current loss value (optional)
        """
        if step % self.log_interval != 0:
            return

        # Compute statistics
        stats = self.compute_gradient_stats(model)

        # Store statistics
        self.steps.append(step)
        self.gradient_norms.append(stats['total_norm'])
        self.max_grads.append(stats['max_grad'])
        self.min_grads.append(stats['min_grad'])
        self.mean_grads.append(stats['mean_grad'])

        # Store per-layer statistics
        if self.track_per_layer:
            for layer_name, layer_stat in stats['layer_stats'].items():
                self.layer_norms[layer_name].append(layer_stat['norm'])

        # Detect issues
        issues = self.detect_issues(stats, step)

        # Print statistics
        print(f"\n{'='*60}")
        print(f"Step {step} - Gradient Statistics")
        print(f"{'='*60}")
        print(f"Total Norm:      {stats['total_norm']:.6f}")
        print(f"Max Gradient:    {stats['max_grad']:.6f}")
        print(f"Min Gradient:    {stats['min_grad']:.6f}")
        print(f"Mean Gradient:   {stats['mean_grad']:.6f}")
        print(f"Params w/ Grad:  {stats['num_params']}")

        if loss is not None:
            print(f"Loss:            {loss:.6f}")

        if optimizer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Learning Rate {i}: {param_group['lr']:.2e}")

        # Print issues
        if issues:
            print(f"\n⚠️  ISSUES DETECTED:")
            for issue in issues:
                print(f"  - {issue}")

        # Print recommendations
        recommendations = self.get_recommendations(stats)
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  - {rec}")

        print(f"{'='*60}\n")

        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar('Gradients/total_norm', stats['total_norm'], step)
            self.writer.add_scalar('Gradients/max_grad', stats['max_grad'], step)
            self.writer.add_scalar('Gradients/min_grad', stats['min_grad'], step)
            self.writer.add_scalar('Gradients/mean_grad', stats['mean_grad'], step)

            if loss is not None:
                self.writer.add_scalar('Loss/train', loss, step)

            # Log per-layer norms
            if self.track_per_layer:
                for layer_name, layer_stat in stats['layer_stats'].items():
                    self.writer.add_scalar(f'Gradients/layer_{layer_name}', layer_stat['norm'], step)

    def save_statistics(self, filename: str = "gradient_stats.json") -> None:
        """
        Save gradient statistics to JSON file

        Args:
            filename: Output filename
        """
        stats_dict = {
            'steps': self.steps,
            'gradient_norms': self.gradient_norms,
            'max_grads': self.max_grads,
            'min_grads': self.min_grads,
            'mean_grads': self.mean_grads,
            'explosions': self.explosions,
            'vanishings': self.vanishings,
            'layer_norms': dict(self.layer_norms),
        }

        output_path = self.log_dir / filename
        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        print(f"Gradient statistics saved to: {output_path}")

    def plot_gradients(self, filename: str = "gradient_plot.png") -> None:
        """
        Create visualization plots of gradient statistics

        Args:
            filename: Output filename for plot
        """
        if not self.save_plots or len(self.steps) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Gradient norm over time
        axes[0, 0].plot(self.steps, self.gradient_norms, linewidth=2, color='blue')
        axes[0, 0].axhline(y=self.explosion_threshold, color='red', linestyle='--', label='Explosion threshold')
        axes[0, 0].axhline(y=self.vanishing_threshold, color='orange', linestyle='--', label='Vanishing threshold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].set_title('Total Gradient Norm')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')

        # Plot 2: Max/Min gradients
        axes[0, 1].plot(self.steps, self.max_grads, label='Max', linewidth=2, color='red')
        axes[0, 1].plot(self.steps, self.min_grads, label='Min', linewidth=2, color='green')
        axes[0, 1].plot(self.steps, self.mean_grads, label='Mean', linewidth=2, color='blue')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Gradient Value')
        axes[0, 1].set_title('Gradient Statistics (Max/Min/Mean)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # Plot 3: Per-layer gradient norms (top layers)
        if self.track_per_layer and self.layer_norms:
            # Get top 5 layers by final gradient norm
            final_norms = {layer: norms[-1] if norms else 0 for layer, norms in self.layer_norms.items()}
            top_layers = sorted(final_norms.items(), key=lambda x: x[1], reverse=True)[:5]

            for layer_name, _ in top_layers:
                if layer_name in self.layer_norms and len(self.layer_norms[layer_name]) > 0:
                    # Ensure same length as steps
                    layer_steps = self.steps[:len(self.layer_norms[layer_name])]
                    axes[1, 0].plot(layer_steps, self.layer_norms[layer_name], label=layer_name, linewidth=2)

            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Layer Gradient Norm')
            axes[1, 0].set_title('Top 5 Layers by Gradient Norm')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        # Plot 4: Gradient norm histogram (last 100 steps)
        recent_norms = self.gradient_norms[-100:] if len(self.gradient_norms) > 100 else self.gradient_norms
        axes[1, 1].hist(recent_norms, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=np.mean(recent_norms), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(recent_norms):.4f}')
        axes[1, 1].set_xlabel('Gradient Norm')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Gradient Norm Distribution (Recent 100 Steps)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.log_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Gradient plot saved to: {output_path}")

    def generate_report(self, filename: str = "gradient_report.txt") -> None:
        """
        Generate text report of gradient statistics

        Args:
            filename: Output filename for report
        """
        if len(self.steps) == 0:
            print("No gradient statistics to report")
            return

        output_path = self.log_dir / filename

        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("GRADIENT MONITORING REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Summary statistics
            f.write("Summary Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total steps monitored:    {len(self.steps)}\n")
            f.write(f"Average gradient norm:    {np.mean(self.gradient_norms):.6f}\n")
            f.write(f"Max gradient norm:        {np.max(self.gradient_norms):.6f}\n")
            f.write(f"Min gradient norm:        {np.min(self.gradient_norms):.6f}\n")
            f.write(f"Std gradient norm:        {np.std(self.gradient_norms):.6f}\n\n")

            # Issues detected
            f.write("Issues Detected:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Gradient explosions:      {len(self.explosions)} (steps: {self.explosions})\n")
            f.write(f"Vanishing gradients:      {len(self.vanishings)} (steps: {self.vanishings})\n\n")

            # Per-layer statistics
            if self.track_per_layer and self.layer_norms:
                f.write("Per-Layer Statistics (Final Values):\n")
                f.write("-" * 60 + "\n")
                for layer_name, norms in sorted(self.layer_norms.items()):
                    if norms:
                        f.write(f"{layer_name:30s}: {norms[-1]:.6f}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"Gradient report saved to: {output_path}")

    def close(self) -> None:
        """Close TensorBoard writer and save final statistics"""
        if self.writer is not None:
            self.writer.close()

        # Save final statistics
        self.save_statistics()
        self.plot_gradients()
        self.generate_report()


# Example usage
if __name__ == "__main__":
    print("Gradient Monitor Module")
    print("=" * 60)

    # Create dummy model for testing
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )

    # Initialize monitor
    monitor = GradientMonitor(
        log_dir="test_gradients",
        log_interval=1,
        enable_tensorboard=False,
    )

    # Simulate training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step in range(10):
        # Forward pass
        x = torch.randn(32, 100)
        y = model(x)
        loss = y.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Log gradients
        monitor.log_gradients(model, optimizer, step, loss.item())

        optimizer.step()

    # Close and save
    monitor.close()

    print("\n" + "=" * 60)
    print("Gradient monitoring complete!")
    print("Check test_gradients/ directory for results")
