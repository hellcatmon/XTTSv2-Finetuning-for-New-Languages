#!/usr/bin/env python3
"""
Comprehensive Training Monitoring Dashboard for XTTS
Real-time monitoring, visualization, and alerting for training runs

Usage:
    from training_dashboard import TrainingDashboard

    dashboard = TrainingDashboard(log_dir="checkpoints/")
    dashboard.log_metrics(step, loss=2.5, lr=5e-6, gpu_memory=18.2)
    dashboard.close()
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import torch
import numpy as np
from collections import defaultdict, deque


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available. Install with: pip install tensorboard")


@dataclass
class TrainingMetrics:
    """Training metrics container"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gpu_memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    samples_per_sec: float = 0.0
    grad_norm: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'gpu_memory_gb': self.gpu_memory_gb,
            'gpu_utilization': self.gpu_utilization,
            'samples_per_sec': self.samples_per_sec,
            'grad_norm': self.grad_norm,
            'timestamp': self.timestamp,
        }


class TrainingDashboard:
    """
    Comprehensive training monitoring dashboard

    Features:
    - Real-time metric tracking
    - TensorBoard integration
    - Automatic alerting for issues
    - Performance profiling
    - Resource monitoring (GPU, memory)
    - Training speed tracking
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        enable_tensorboard: bool = True,
        log_interval: int = 100,
        save_interval: int = 1000,
        alert_thresholds: Optional[Dict] = None,
        max_history: int = 10000,
    ):
        """
        Initialize training dashboard

        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment (default: timestamp)
            enable_tensorboard: Enable TensorBoard logging
            log_interval: Log metrics every N steps
            save_interval: Save metrics to disk every N steps
            alert_thresholds: Thresholds for alerting (dict with metric: threshold)
            max_history: Maximum number of metrics to keep in memory
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.max_history = max_history

        # Create log directory
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard
        self.writer = None
        if self.enable_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.experiment_dir))
            print(f"📊 TensorBoard logging enabled: {self.experiment_dir}")
            print(f"   View with: tensorboard --logdir {self.log_dir}")

        # Metric history (using deque for memory efficiency)
        self.metrics_history = deque(maxlen=max_history)

        # Current metrics
        self.current_metrics = TrainingMetrics()

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'loss': 10.0,  # Alert if loss > 10
            'grad_norm': 100.0,  # Alert if grad norm > 100
            'gpu_memory_gb': 22.0,  # Alert if memory > 22GB
        }

        # Performance tracking
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.step_times = deque(maxlen=100)

        # Alert history
        self.alerts = []

        # Statistics
        self.stats = {
            'best_loss': float('inf'),
            'best_step': 0,
            'total_steps': 0,
            'total_samples': 0,
            'avg_samples_per_sec': 0.0,
        }

        print(f"✅ Training dashboard initialized")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Log directory: {self.experiment_dir}")

    def log_metrics(
        self,
        step: int,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        gpu_memory: Optional[float] = None,
        gpu_util: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
        custom_metrics: Optional[Dict] = None,
    ):
        """
        Log training metrics

        Args:
            step: Training step
            epoch: Current epoch
            loss: Training loss
            lr: Learning rate
            grad_norm: Gradient norm
            gpu_memory: GPU memory usage in GB
            gpu_util: GPU utilization percentage
            samples_per_sec: Samples processed per second
            custom_metrics: Additional custom metrics
        """
        # Update current metrics
        self.current_metrics.step = step
        if epoch is not None:
            self.current_metrics.epoch = epoch
        if loss is not None:
            self.current_metrics.loss = loss
        if lr is not None:
            self.current_metrics.learning_rate = lr
        if grad_norm is not None:
            self.current_metrics.grad_norm = grad_norm
        if gpu_memory is not None:
            self.current_metrics.gpu_memory_gb = gpu_memory
        if gpu_util is not None:
            self.current_metrics.gpu_utilization = gpu_util
        if samples_per_sec is not None:
            self.current_metrics.samples_per_sec = samples_per_sec

        # Calculate step time
        current_time = time.time()
        step_time = current_time - self.last_log_time
        self.step_times.append(step_time)
        self.last_log_time = current_time

        # Log to TensorBoard
        if self.enable_tensorboard and step % self.log_interval == 0:
            if loss is not None:
                self.writer.add_scalar('Loss/train', loss, step)
            if lr is not None:
                self.writer.add_scalar('LearningRate', lr, step)
            if grad_norm is not None:
                self.writer.add_scalar('Gradients/norm', grad_norm, step)
            if gpu_memory is not None:
                self.writer.add_scalar('Resources/gpu_memory_gb', gpu_memory, step)
            if gpu_util is not None:
                self.writer.add_scalar('Resources/gpu_utilization', gpu_util, step)
            if samples_per_sec is not None:
                self.writer.add_scalar('Performance/samples_per_sec', samples_per_sec, step)

            # Log custom metrics
            if custom_metrics:
                for key, value in custom_metrics.items():
                    self.writer.add_scalar(f'Custom/{key}', value, step)

        # Add to history
        self.metrics_history.append(self.current_metrics.to_dict())

        # Update statistics
        self._update_statistics()

        # Check for alerts
        self._check_alerts()

        # Save metrics periodically
        if step % self.save_interval == 0:
            self._save_metrics()

        # Print progress
        if step % self.log_interval == 0:
            self._print_progress()

    def log_model_outputs(
        self,
        step: int,
        audio: torch.Tensor,
        sample_rate: int = 22050,
        tag: str = "audio",
    ):
        """
        Log audio samples to TensorBoard

        Args:
            step: Training step
            audio: Audio tensor (shape: [channels, samples])
            sample_rate: Sample rate
            tag: Tag for the audio
        """
        if self.enable_tensorboard:
            self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)

    def log_text_sample(
        self,
        step: int,
        text: str,
        tag: str = "text",
    ):
        """
        Log text sample to TensorBoard

        Args:
            step: Training step
            text: Text to log
            tag: Tag for the text
        """
        if self.enable_tensorboard:
            self.writer.add_text(tag, text, step)

    def log_histogram(
        self,
        step: int,
        values: torch.Tensor,
        tag: str,
    ):
        """
        Log histogram to TensorBoard

        Args:
            step: Training step
            values: Values to plot
            tag: Tag for the histogram
        """
        if self.enable_tensorboard:
            self.writer.add_histogram(tag, values, step)

    def log_image(
        self,
        step: int,
        image: torch.Tensor,
        tag: str,
    ):
        """
        Log image to TensorBoard

        Args:
            step: Training step
            image: Image tensor
            tag: Tag for the image
        """
        if self.enable_tensorboard:
            self.writer.add_image(tag, image, step)

    def _update_statistics(self):
        """Update training statistics"""
        # Update best loss
        if self.current_metrics.loss < self.stats['best_loss']:
            self.stats['best_loss'] = self.current_metrics.loss
            self.stats['best_step'] = self.current_metrics.step

        # Update total steps
        self.stats['total_steps'] = self.current_metrics.step

        # Update average samples per second
        if self.current_metrics.samples_per_sec > 0:
            # Exponential moving average
            alpha = 0.1
            self.stats['avg_samples_per_sec'] = (
                alpha * self.current_metrics.samples_per_sec +
                (1 - alpha) * self.stats['avg_samples_per_sec']
            )

    def _check_alerts(self):
        """Check for alert conditions"""
        alerts = []

        for metric, threshold in self.alert_thresholds.items():
            value = getattr(self.current_metrics, metric, None)
            if value is not None and value > threshold:
                alert = f"⚠️  {metric} = {value:.2f} exceeds threshold {threshold}"
                alerts.append(alert)

        # Check for NaN loss
        if self.current_metrics.loss != self.current_metrics.loss:  # NaN check
            alerts.append("❌ NaN loss detected!")

        # Check for loss explosion
        if len(self.metrics_history) > 10:
            recent_losses = [m['loss'] for m in list(self.metrics_history)[-10:] if m['loss'] > 0]
            if recent_losses and self.current_metrics.loss > np.mean(recent_losses) * 5:
                alerts.append("⚠️  Loss explosion detected!")

        # Store and print alerts
        for alert in alerts:
            self.alerts.append({
                'step': self.current_metrics.step,
                'message': alert,
                'timestamp': time.time(),
            })
            print(f"\n{alert}\n")

    def _print_progress(self):
        """Print training progress"""
        elapsed_time = time.time() - self.start_time
        avg_step_time = np.mean(self.step_times) if self.step_times else 0

        # Estimate remaining time (rough estimate)
        # This is a simple estimate - actual remaining time depends on total steps

        print(f"[Step {self.current_metrics.step}] "
              f"Loss: {self.current_metrics.loss:.4f} | "
              f"LR: {self.current_metrics.learning_rate:.2e} | "
              f"GradNorm: {self.current_metrics.grad_norm:.4f} | "
              f"GPU: {self.current_metrics.gpu_memory_gb:.1f}GB "
              f"({self.current_metrics.gpu_utilization:.0f}%) | "
              f"Speed: {self.current_metrics.samples_per_sec:.1f} samples/s | "
              f"Time: {avg_step_time:.2f}s/step")

    def _save_metrics(self):
        """Save metrics to disk"""
        metrics_file = self.experiment_dir / "metrics.jsonl"

        # Save current metrics
        with open(metrics_file, 'a') as f:
            json.dump(self.current_metrics.to_dict(), f)
            f.write('\n')

        # Save statistics
        stats_file = self.experiment_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Save alerts
        if self.alerts:
            alerts_file = self.experiment_dir / "alerts.json"
            with open(alerts_file, 'w') as f:
                json.dump(self.alerts, f, indent=2)

    def get_summary(self) -> Dict:
        """
        Get training summary

        Returns:
            Dictionary with training summary
        """
        elapsed_time = time.time() - self.start_time

        return {
            'experiment_name': self.experiment_name,
            'current_step': self.current_metrics.step,
            'current_loss': self.current_metrics.loss,
            'best_loss': self.stats['best_loss'],
            'best_step': self.stats['best_step'],
            'elapsed_time_hours': elapsed_time / 3600,
            'avg_samples_per_sec': self.stats['avg_samples_per_sec'],
            'total_alerts': len(self.alerts),
        }

    def print_summary(self):
        """Print training summary"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)
        print(f"Experiment: {summary['experiment_name']}")
        print(f"Current Step: {summary['current_step']}")
        print(f"Current Loss: {summary['current_loss']:.4f}")
        print(f"Best Loss: {summary['best_loss']:.4f} (step {summary['best_step']})")
        print(f"Elapsed Time: {summary['elapsed_time_hours']:.2f} hours")
        print(f"Avg Speed: {summary['avg_samples_per_sec']:.1f} samples/sec")
        print(f"Total Alerts: {summary['total_alerts']}")
        print("=" * 80 + "\n")

    def close(self):
        """Close dashboard and save final metrics"""
        # Save final metrics
        self._save_metrics()

        # Print summary
        self.print_summary()

        # Close TensorBoard
        if self.writer:
            self.writer.close()

        print(f"✅ Dashboard closed. Logs saved to: {self.experiment_dir}")


class PerformanceProfiler:
    """
    Performance profiler for training bottleneck detection
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize profiler

        Args:
            enabled: Enable profiling
        """
        self.enabled = enabled
        self.timings = defaultdict(list)
        self.current_context = None
        self.start_time = None

    def __enter__(self):
        """Context manager entry"""
        if self.enabled:
            self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """Context manager exit"""
        if self.enabled and self.start_time and self.current_context:
            elapsed = time.time() - self.start_time
            self.timings[self.current_context].append(elapsed)

    def profile(self, name: str):
        """
        Profile a code section

        Usage:
            profiler = PerformanceProfiler()
            with profiler.profile("data_loading"):
                # code to profile
                pass
        """
        self.current_context = name
        return self

    def get_summary(self) -> Dict:
        """Get profiling summary"""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'total_time': np.sum(times),
                'count': len(times),
            }
        return summary

    def print_summary(self):
        """Print profiling summary"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("Performance Profile")
        print("=" * 80)

        # Sort by total time
        sorted_items = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)

        for name, stats in sorted_items:
            print(f"{name}:")
            print(f"  Average: {stats['avg_time']*1000:.2f}ms")
            print(f"  Total: {stats['total_time']:.2f}s")
            print(f"  Count: {stats['count']}")

        print("=" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    print("Training Dashboard Module")
    print("=" * 80)

    # Create dashboard
    dashboard = TrainingDashboard(
        log_dir="logs",
        experiment_name="test_run",
        enable_tensorboard=TENSORBOARD_AVAILABLE,
    )

    # Simulate training
    print("\nSimulating training steps...")
    for step in range(1, 1001):
        # Simulate metrics
        loss = 5.0 * (0.995 ** step) + np.random.rand() * 0.1
        lr = 5e-6
        grad_norm = 1.0 + np.random.rand() * 0.5
        gpu_memory = 18.0 + np.random.rand() * 2
        gpu_util = 90 + np.random.rand() * 10
        samples_per_sec = 100 + np.random.rand() * 20

        # Log metrics
        dashboard.log_metrics(
            step=step,
            epoch=step // 100,
            loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            gpu_memory=gpu_memory,
            gpu_util=gpu_util,
            samples_per_sec=samples_per_sec,
        )

        time.sleep(0.01)  # Simulate training time

    # Close dashboard
    dashboard.close()

    print("\n" + "=" * 80)
    print("Dashboard module loaded successfully!")
    print("Use: from training_dashboard import TrainingDashboard")
