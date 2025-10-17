#!/usr/bin/env python3
"""
DeepSpeed Configuration for Multi-GPU XTTS Training
Provides optimized DeepSpeed configurations for distributed training

Usage:
    from deepspeed_config import get_deepspeed_config, DeepSpeedPreset

    config = get_deepspeed_config(DeepSpeedPreset.ZERO2_FP16)
    # Use with deepspeed launcher:
    # deepspeed train_gpt_xtts.py --deepspeed --deepspeed_config ds_config.json
"""

import json
from enum import Enum
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict


class DeepSpeedPreset(Enum):
    """DeepSpeed configuration presets"""
    ZERO0 = "zero0"  # No ZeRO optimization (baseline distributed)
    ZERO1 = "zero1"  # ZeRO Stage 1 (optimizer state partitioning)
    ZERO2 = "zero2"  # ZeRO Stage 2 (optimizer + gradient partitioning)
    ZERO3 = "zero3"  # ZeRO Stage 3 (optimizer + gradient + parameter partitioning)
    ZERO2_OFFLOAD = "zero2_offload"  # ZeRO Stage 2 with CPU offload
    ZERO3_OFFLOAD = "zero3_offload"  # ZeRO Stage 3 with CPU offload


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration builder"""

    # Training settings
    train_batch_size: int = 32
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 8
    steps_per_print: int = 100

    # Optimizer
    optimizer_type: str = "AdamW"
    optimizer_params: Dict = None

    # Learning rate scheduler
    scheduler_type: str = "WarmupDecayLR"
    scheduler_params: Dict = None

    # Mixed precision
    fp16_enabled: bool = False
    fp16_loss_scale: float = 0.0
    fp16_initial_scale_power: int = 16
    fp16_loss_scale_window: int = 1000
    fp16_hysteresis: int = 2
    fp16_min_loss_scale: float = 1.0

    bf16_enabled: bool = False

    # ZeRO optimization
    zero_stage: int = 0
    zero_offload_optimizer: bool = False
    zero_offload_optimizer_device: str = "cpu"
    zero_offload_optimizer_pin_memory: bool = True
    zero_offload_param: bool = False
    zero_offload_param_device: str = "cpu"
    zero_offload_param_pin_memory: bool = True
    zero_reduce_scatter: bool = True
    zero_allgather_partitions: bool = True
    zero_allgather_bucket_size: int = 500000000
    zero_reduce_bucket_size: int = 500000000
    zero_overlap_comm: bool = True
    zero_contiguous_gradients: bool = True

    # Gradient clipping
    gradient_clipping: float = 1.0

    # Activation checkpointing
    activation_checkpointing_enabled: bool = False
    activation_checkpointing_partition_activations: bool = False
    activation_checkpointing_number_checkpoints: Optional[int] = None

    # Wall clock breakdown
    wall_clock_breakdown: bool = False

    # Additional settings
    prescale_gradients: bool = False
    sparse_gradients: bool = False

    def __post_init__(self):
        """Set default optimizer and scheduler params"""
        if self.optimizer_params is None:
            self.optimizer_params = {
                "lr": 5e-6,
                "betas": [0.9, 0.96],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }

        if self.scheduler_params is None:
            self.scheduler_params = {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": 5e-6,
                "warmup_num_steps": 1000,
                "total_num_steps": 100000,
            }

    def to_dict(self) -> Dict:
        """Convert to DeepSpeed JSON config format"""
        config = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_print": self.steps_per_print,
            "gradient_clipping": self.gradient_clipping,
            "prescale_gradients": self.prescale_gradients,
            "wall_clock_breakdown": self.wall_clock_breakdown,
        }

        # Optimizer
        config["optimizer"] = {
            "type": self.optimizer_type,
            "params": self.optimizer_params,
        }

        # Scheduler
        config["scheduler"] = {
            "type": self.scheduler_type,
            "params": self.scheduler_params,
        }

        # FP16 mixed precision
        if self.fp16_enabled:
            config["fp16"] = {
                "enabled": True,
                "loss_scale": self.fp16_loss_scale,
                "initial_scale_power": self.fp16_initial_scale_power,
                "loss_scale_window": self.fp16_loss_scale_window,
                "hysteresis": self.fp16_hysteresis,
                "min_loss_scale": self.fp16_min_loss_scale,
            }

        # BF16 mixed precision
        if self.bf16_enabled:
            config["bf16"] = {
                "enabled": True,
            }

        # ZeRO optimization
        if self.zero_stage > 0:
            zero_config = {
                "stage": self.zero_stage,
                "reduce_scatter": self.zero_reduce_scatter,
                "allgather_partitions": self.zero_allgather_partitions,
                "allgather_bucket_size": self.zero_allgather_bucket_size,
                "reduce_bucket_size": self.zero_reduce_bucket_size,
                "overlap_comm": self.zero_overlap_comm,
                "contiguous_gradients": self.zero_contiguous_gradients,
            }

            # Optimizer offload
            if self.zero_offload_optimizer:
                zero_config["offload_optimizer"] = {
                    "device": self.zero_offload_optimizer_device,
                    "pin_memory": self.zero_offload_optimizer_pin_memory,
                }

            # Parameter offload (ZeRO-3 only)
            if self.zero_stage == 3 and self.zero_offload_param:
                zero_config["offload_param"] = {
                    "device": self.zero_offload_param_device,
                    "pin_memory": self.zero_offload_param_pin_memory,
                }

            config["zero_optimization"] = zero_config

        # Activation checkpointing
        if self.activation_checkpointing_enabled:
            ac_config = {
                "partition_activations": self.activation_checkpointing_partition_activations,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
                "synchronize_checkpoint_boundary": False,
            }
            if self.activation_checkpointing_number_checkpoints:
                ac_config["number_checkpoints"] = self.activation_checkpointing_number_checkpoints

            config["activation_checkpointing"] = ac_config

        return config

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ DeepSpeed config saved to: {filepath}")


def get_deepspeed_config(
    preset: DeepSpeedPreset = DeepSpeedPreset.ZERO2,
    train_batch_size: int = 32,
    micro_batch_size: int = 4,
    learning_rate: float = 5e-6,
    total_steps: int = 100000,
    use_bf16: bool = False,
    use_fp16: bool = True,
) -> DeepSpeedConfig:
    """
    Get DeepSpeed configuration with preset

    Args:
        preset: DeepSpeed preset configuration
        train_batch_size: Total training batch size across all GPUs
        micro_batch_size: Batch size per GPU
        learning_rate: Learning rate
        total_steps: Total training steps
        use_bf16: Use BFloat16 (Ampere+ GPUs)
        use_fp16: Use FP16 (older GPUs)

    Returns:
        DeepSpeedConfig object
    """
    gradient_accumulation = train_batch_size // micro_batch_size

    # Base config
    config = DeepSpeedConfig(
        train_batch_size=train_batch_size,
        train_micro_batch_size_per_gpu=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
    )

    # Update optimizer learning rate
    config.optimizer_params["lr"] = learning_rate
    config.scheduler_params["warmup_max_lr"] = learning_rate
    config.scheduler_params["total_num_steps"] = total_steps

    # Apply preset
    if preset == DeepSpeedPreset.ZERO0:
        # No ZeRO optimization
        config.zero_stage = 0
        config.fp16_enabled = use_fp16
        config.bf16_enabled = use_bf16

    elif preset == DeepSpeedPreset.ZERO1:
        # ZeRO Stage 1: Optimizer state partitioning
        config.zero_stage = 1
        config.fp16_enabled = use_fp16
        config.bf16_enabled = use_bf16

    elif preset == DeepSpeedPreset.ZERO2:
        # ZeRO Stage 2: Optimizer + gradient partitioning
        config.zero_stage = 2
        config.fp16_enabled = use_fp16
        config.bf16_enabled = use_bf16
        config.zero_reduce_scatter = True
        config.zero_allgather_partitions = True
        config.zero_overlap_comm = True
        config.zero_contiguous_gradients = True

    elif preset == DeepSpeedPreset.ZERO3:
        # ZeRO Stage 3: Optimizer + gradient + parameter partitioning
        config.zero_stage = 3
        config.fp16_enabled = use_fp16
        config.bf16_enabled = use_bf16
        config.zero_reduce_scatter = True
        config.zero_allgather_partitions = True
        config.zero_overlap_comm = True
        config.zero_contiguous_gradients = True

    elif preset == DeepSpeedPreset.ZERO2_OFFLOAD:
        # ZeRO Stage 2 with CPU offload
        config.zero_stage = 2
        config.fp16_enabled = use_fp16
        config.bf16_enabled = use_bf16
        config.zero_offload_optimizer = True
        config.zero_offload_optimizer_device = "cpu"
        config.zero_offload_optimizer_pin_memory = True
        config.zero_reduce_scatter = True
        config.zero_allgather_partitions = True
        config.zero_overlap_comm = True
        config.zero_contiguous_gradients = True

    elif preset == DeepSpeedPreset.ZERO3_OFFLOAD:
        # ZeRO Stage 3 with CPU offload
        config.zero_stage = 3
        config.fp16_enabled = use_fp16
        config.bf16_enabled = use_bf16
        config.zero_offload_optimizer = True
        config.zero_offload_optimizer_device = "cpu"
        config.zero_offload_optimizer_pin_memory = True
        config.zero_offload_param = True
        config.zero_offload_param_device = "cpu"
        config.zero_offload_param_pin_memory = True
        config.zero_reduce_scatter = True
        config.zero_allgather_partitions = True
        config.zero_overlap_comm = True
        config.zero_contiguous_gradients = True

    return config


def create_all_presets(output_dir: str = "deepspeed_configs"):
    """
    Create all DeepSpeed preset configurations

    Args:
        output_dir: Directory to save config files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Creating DeepSpeed Configuration Presets")
    print("=" * 80)

    for preset in DeepSpeedPreset:
        print(f"\nCreating preset: {preset.value}")

        # Create config
        config = get_deepspeed_config(
            preset=preset,
            train_batch_size=32,
            micro_batch_size=4,
            learning_rate=5e-6,
            total_steps=100000,
            use_bf16=False,
            use_fp16=True,
        )

        # Save config
        filename = output_path / f"ds_config_{preset.value}.json"
        config.save(str(filename))

        # Print summary
        config_dict = config.to_dict()
        print(f"  - Train batch size: {config_dict['train_batch_size']}")
        print(f"  - Micro batch size: {config_dict['train_micro_batch_size_per_gpu']}")
        print(f"  - Gradient accumulation: {config_dict['gradient_accumulation_steps']}")
        if 'zero_optimization' in config_dict:
            print(f"  - ZeRO stage: {config_dict['zero_optimization']['stage']}")
            if 'offload_optimizer' in config_dict['zero_optimization']:
                print(f"  - Optimizer offload: CPU")
            if 'offload_param' in config_dict['zero_optimization']:
                print(f"  - Parameter offload: CPU")
        print(f"  - Mixed precision: {'BF16' if config.bf16_enabled else 'FP16' if config.fp16_enabled else 'FP32'}")

    print("\n" + "=" * 80)
    print(f"All configs saved to: {output_path}")
    print("=" * 80)


def get_recommended_preset(
    num_gpus: int,
    gpu_memory_gb: int,
    model_size_gb: float = 3.0,
) -> DeepSpeedPreset:
    """
    Get recommended DeepSpeed preset based on hardware

    Args:
        num_gpus: Number of GPUs
        gpu_memory_gb: GPU memory in GB
        model_size_gb: Model size in GB

    Returns:
        Recommended DeepSpeed preset
    """
    print(f"\n{'='*60}")
    print(f"DeepSpeed Preset Recommendation")
    print(f"{'='*60}")
    print(f"GPUs: {num_gpus}")
    print(f"GPU Memory: {gpu_memory_gb} GB")
    print(f"Model Size: {model_size_gb} GB")

    # Calculate memory requirements
    model_memory = model_size_gb
    optimizer_memory = model_memory * 2  # Adam states
    gradient_memory = model_memory
    activation_memory = model_memory * 2  # Rough estimate

    total_memory_required = model_memory + optimizer_memory + gradient_memory + activation_memory

    print(f"\nEstimated memory requirements:")
    print(f"  Model: {model_memory:.1f} GB")
    print(f"  Optimizer: {optimizer_memory:.1f} GB")
    print(f"  Gradients: {gradient_memory:.1f} GB")
    print(f"  Activations: {activation_memory:.1f} GB")
    print(f"  Total: {total_memory_required:.1f} GB")

    # Available memory across all GPUs
    available_memory = num_gpus * gpu_memory_gb * 0.85  # 85% usable
    print(f"\nAvailable GPU memory: {available_memory:.1f} GB")

    # Recommend preset
    if available_memory >= total_memory_required:
        preset = DeepSpeedPreset.ZERO1
        print(f"\n✅ Recommendation: {preset.value}")
        print("   Sufficient memory for ZeRO-1 (optimizer state partitioning)")

    elif available_memory >= model_memory + gradient_memory + activation_memory:
        preset = DeepSpeedPreset.ZERO2
        print(f"\n✅ Recommendation: {preset.value}")
        print("   Use ZeRO-2 (optimizer + gradient partitioning)")

    elif available_memory >= model_memory + activation_memory:
        preset = DeepSpeedPreset.ZERO3
        print(f"\n✅ Recommendation: {preset.value}")
        print("   Use ZeRO-3 (full parameter partitioning)")

    elif gpu_memory_gb * num_gpus >= 24:  # At least 24GB total
        preset = DeepSpeedPreset.ZERO2_OFFLOAD
        print(f"\n✅ Recommendation: {preset.value}")
        print("   Use ZeRO-2 with CPU offload")

    else:
        preset = DeepSpeedPreset.ZERO3_OFFLOAD
        print(f"\n✅ Recommendation: {preset.value}")
        print("   Use ZeRO-3 with full CPU offload")

    print(f"{'='*60}\n")

    return preset


# Example usage
if __name__ == "__main__":
    print("DeepSpeed Configuration Module")
    print("=" * 80)

    # Create all preset configs
    create_all_presets("deepspeed_configs")

    # Test recommendations
    print("\n" + "=" * 80)
    print("Hardware-based Recommendations")
    print("=" * 80)

    # Example scenarios
    scenarios = [
        {"num_gpus": 2, "gpu_memory_gb": 24, "description": "2x RTX 3090/4090"},
        {"num_gpus": 4, "gpu_memory_gb": 16, "description": "4x RTX 4080"},
        {"num_gpus": 2, "gpu_memory_gb": 12, "description": "2x RTX 3060"},
        {"num_gpus": 8, "gpu_memory_gb": 40, "description": "8x A100"},
    ]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['description']}")
        preset = get_recommended_preset(
            num_gpus=scenario['num_gpus'],
            gpu_memory_gb=scenario['gpu_memory_gb'],
            model_size_gb=3.0,
        )

    print("\n" + "=" * 80)
    print("DeepSpeed configuration module loaded successfully!")
    print("Use: from deepspeed_config import get_deepspeed_config, DeepSpeedPreset")
    print("\nTo train with DeepSpeed:")
    print("  deepspeed train_gpt_xtts.py --deepspeed --deepspeed_config ds_config.json")
