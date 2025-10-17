#!/usr/bin/env python3
"""
Gradient Checkpointing Module for XTTS Training
Reduces memory usage by trading compute for memory during backpropagation

Usage:
    from gradient_checkpointing import apply_gradient_checkpointing

    model = XttsModel(...)
    apply_gradient_checkpointing(model, checkpoint_every_n_layers=2)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List
import warnings


class GradientCheckpointConfig:
    """Configuration for gradient checkpointing"""

    def __init__(
        self,
        enabled: bool = True,
        checkpoint_every_n_layers: int = 1,
        checkpoint_gpt: bool = True,
        checkpoint_dvae: bool = False,
        use_reentrant: bool = False,  # PyTorch 2.0+ supports non-reentrant
    ):
        """
        Initialize gradient checkpointing configuration

        Args:
            enabled: Enable gradient checkpointing
            checkpoint_every_n_layers: Checkpoint every N transformer layers (1=all, 2=every other)
            checkpoint_gpt: Apply checkpointing to GPT model
            checkpoint_dvae: Apply checkpointing to DVAE encoder
            use_reentrant: Use reentrant checkpointing (legacy, not recommended for PyTorch 2.0+)
        """
        self.enabled = enabled
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        self.checkpoint_gpt = checkpoint_gpt
        self.checkpoint_dvae = checkpoint_dvae
        self.use_reentrant = use_reentrant


def apply_gradient_checkpointing(
    model: nn.Module,
    config: Optional[GradientCheckpointConfig] = None,
    checkpoint_every_n_layers: int = 1,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply gradient checkpointing to XTTS model

    Args:
        model: XTTS model (XttsModel or GPT model)
        config: Gradient checkpointing configuration (optional)
        checkpoint_every_n_layers: Checkpoint every N layers if config not provided
        verbose: Print information about checkpointing

    Returns:
        Model with gradient checkpointing enabled
    """
    if config is None:
        config = GradientCheckpointConfig(
            enabled=True,
            checkpoint_every_n_layers=checkpoint_every_n_layers,
        )

    if not config.enabled:
        if verbose:
            print("⚠️  Gradient checkpointing disabled")
        return model

    # Check PyTorch version for non-reentrant checkpointing
    pytorch_version = torch.__version__.split('+')[0]
    major, minor = map(int, pytorch_version.split('.')[:2])
    supports_non_reentrant = (major > 2) or (major == 2 and minor >= 0)

    if not config.use_reentrant and not supports_non_reentrant:
        warnings.warn(
            "Non-reentrant checkpointing requires PyTorch 2.0+. "
            "Falling back to reentrant mode."
        )
        config.use_reentrant = True

    checkpointed_layers = 0

    # Apply to GPT model
    if config.checkpoint_gpt:
        if hasattr(model, 'gpt'):
            gpt_model = model.gpt
        elif hasattr(model, 'xtts') and hasattr(model.xtts, 'gpt'):
            gpt_model = model.xtts.gpt
        else:
            gpt_model = model  # Assume model is GPT itself

        checkpointed_layers += _checkpoint_transformer_layers(
            gpt_model,
            config.checkpoint_every_n_layers,
            config.use_reentrant,
            verbose=verbose,
        )

    # Apply to DVAE encoder
    if config.checkpoint_dvae:
        if hasattr(model, 'dvae'):
            dvae_model = model.dvae
        elif hasattr(model, 'xtts') and hasattr(model.xtts, 'dvae'):
            dvae_model = model.xtts.dvae
        else:
            dvae_model = None

        if dvae_model is not None:
            checkpointed_layers += _checkpoint_dvae_layers(
                dvae_model,
                config.use_reentrant,
                verbose=verbose,
            )

    if verbose:
        memory_saved = checkpointed_layers * 15  # Rough estimate: 15% per layer
        print(f"✅ Gradient checkpointing enabled")
        print(f"   - Checkpointed layers: {checkpointed_layers}")
        print(f"   - Est. memory savings: ~{memory_saved}%")
        print(f"   - Compute overhead: ~20-30%")
        print(f"   - Mode: {'non-reentrant' if not config.use_reentrant else 'reentrant'}")

    return model


def _checkpoint_transformer_layers(
    model: nn.Module,
    checkpoint_every_n: int,
    use_reentrant: bool,
    verbose: bool = True,
) -> int:
    """
    Apply gradient checkpointing to transformer layers

    Returns:
        Number of layers with checkpointing enabled
    """
    checkpointed = 0

    # Try different common transformer layer names
    layer_names = ['layers', 'h', 'blocks', 'transformer_blocks']

    for layer_name in layer_names:
        if hasattr(model, layer_name):
            layers = getattr(model, layer_name)

            if isinstance(layers, nn.ModuleList):
                for i, layer in enumerate(layers):
                    if i % checkpoint_every_n == 0:
                        _enable_checkpointing_for_layer(layer, use_reentrant)
                        checkpointed += 1

                if verbose:
                    print(f"   - GPT: {checkpointed}/{len(layers)} layers checkpointed")
                return checkpointed

    # Fallback: try to enable on entire model
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        checkpointed = 1
        if verbose:
            print(f"   - GPT: enabled via gradient_checkpointing_enable()")

    return checkpointed


def _checkpoint_dvae_layers(
    model: nn.Module,
    use_reentrant: bool,
    verbose: bool = True,
) -> int:
    """
    Apply gradient checkpointing to DVAE encoder layers

    Returns:
        Number of layers with checkpointing enabled
    """
    checkpointed = 0

    # DVAE typically has encoder and decoder
    if hasattr(model, 'encoder'):
        encoder = model.encoder

        # Find encoder blocks
        for name, module in encoder.named_children():
            if 'block' in name.lower() or 'layer' in name.lower():
                _enable_checkpointing_for_layer(module, use_reentrant)
                checkpointed += 1

    if verbose and checkpointed > 0:
        print(f"   - DVAE: {checkpointed} layers checkpointed")

    return checkpointed


def _enable_checkpointing_for_layer(layer: nn.Module, use_reentrant: bool):
    """
    Enable gradient checkpointing for a single layer
    """
    # Wrap the layer's forward method with checkpoint
    original_forward = layer.forward

    def checkpointed_forward(*args, **kwargs):
        def custom_forward(*inputs):
            return original_forward(*inputs)

        # Use torch.utils.checkpoint
        return torch.utils.checkpoint.checkpoint(
            custom_forward,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )

    layer.forward = checkpointed_forward


def estimate_memory_savings(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    num_checkpointed_layers: int,
) -> dict:
    """
    Estimate memory savings from gradient checkpointing

    Args:
        model: The model
        batch_size: Training batch size
        sequence_length: Sequence length
        num_checkpointed_layers: Number of layers with checkpointing

    Returns:
        Dictionary with memory estimates
    """
    # Rough estimates based on typical transformer memory usage
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)

    # Activation memory (rough estimate)
    # Each transformer layer stores: attention outputs, FFN outputs, residuals
    hidden_size = 1024  # Typical for XTTS GPT
    activation_per_layer = batch_size * sequence_length * hidden_size * 4 / (1024**3)  # 4 bytes per float32

    # Total activation memory without checkpointing
    total_activation_memory = activation_per_layer * num_checkpointed_layers * 3  # 3 tensors per layer

    # With checkpointing, we save ~70% of activation memory
    memory_saved = total_activation_memory * 0.7

    return {
        "parameter_memory_gb": param_memory,
        "activation_memory_without_checkpoint_gb": total_activation_memory,
        "activation_memory_with_checkpoint_gb": total_activation_memory - memory_saved,
        "memory_saved_gb": memory_saved,
        "memory_saved_percent": (memory_saved / (param_memory + total_activation_memory)) * 100,
    }


def compare_memory_usage(
    model: nn.Module,
    sample_input: dict,
    use_checkpoint: bool = True,
) -> dict:
    """
    Compare memory usage with and without gradient checkpointing

    Args:
        model: Model to test
        sample_input: Sample input dictionary
        use_checkpoint: Whether to enable checkpointing

    Returns:
        Dictionary with memory usage statistics
    """
    import torch.cuda as cuda

    if not cuda.is_available():
        raise RuntimeError("CUDA not available, cannot measure memory usage")

    # Clear cache
    cuda.empty_cache()
    cuda.reset_peak_memory_stats()

    # Test forward + backward pass
    model.train()

    if use_checkpoint:
        apply_gradient_checkpointing(model, verbose=False)

    # Forward pass
    outputs = model(**sample_input)
    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

    # Backward pass
    loss.backward()

    # Measure memory
    memory_allocated = cuda.memory_allocated() / (1024**3)
    memory_reserved = cuda.memory_reserved() / (1024**3)
    max_memory_allocated = cuda.max_memory_allocated() / (1024**3)

    # Clean up
    model.zero_grad()
    cuda.empty_cache()

    return {
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "max_memory_allocated_gb": max_memory_allocated,
        "checkpoint_enabled": use_checkpoint,
    }


# Example usage and testing
if __name__ == "__main__":
    print("Gradient Checkpointing Module")
    print("=" * 60)

    # Create test configuration
    configs = {
        "aggressive": GradientCheckpointConfig(
            enabled=True,
            checkpoint_every_n_layers=1,  # All layers
            checkpoint_gpt=True,
            checkpoint_dvae=False,
        ),
        "balanced": GradientCheckpointConfig(
            enabled=True,
            checkpoint_every_n_layers=2,  # Every other layer
            checkpoint_gpt=True,
            checkpoint_dvae=False,
        ),
        "conservative": GradientCheckpointConfig(
            enabled=True,
            checkpoint_every_n_layers=4,  # Every 4th layer
            checkpoint_gpt=True,
            checkpoint_dvae=False,
        ),
    }

    print("\nAvailable Configurations:")
    for name, config in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  - Checkpoint every: {config.checkpoint_every_n_layers} layers")
        print(f"  - GPT: {config.checkpoint_gpt}")
        print(f"  - DVAE: {config.checkpoint_dvae}")
        print(f"  - Use reentrant: {config.use_reentrant}")

    # Memory savings estimation
    print("\n" + "=" * 60)
    print("Memory Savings Estimation (example):")
    print("=" * 60)

    # Example: GPT model with 24 layers
    num_layers = 24
    batch_size = 8
    seq_length = 250

    for name, config in configs.items():
        checkpointed_layers = num_layers // config.checkpoint_every_n_layers
        print(f"\n{name.upper()} (checkpoint every {config.checkpoint_every_n_layers} layers):")
        print(f"  - Checkpointed layers: {checkpointed_layers}/{num_layers}")
        print(f"  - Est. memory savings: ~{checkpointed_layers * 15}%")
        print(f"  - Est. compute overhead: ~20-30%")

    print("\n" + "=" * 60)
    print("Gradient checkpointing module loaded successfully!")
    print("Use: from gradient_checkpointing import apply_gradient_checkpointing")
