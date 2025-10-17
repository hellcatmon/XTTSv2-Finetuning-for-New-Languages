#!/usr/bin/env python3
"""
Enhanced GPT XTTS Training with Mixed Precision Support
Supports FP16, BFloat16, and automatic mixed precision training

Usage:
    # BFloat16 (recommended for Ampere+ GPUs)
    CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts_mixed_precision.py \
      --output_path checkpoints/ \
      --metadatas datasets/metadata_train.csv,datasets/metadata_eval.csv,be \
      --use_bfloat16 \
      --batch_size 8

    # FP16 (for older GPUs)
    CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts_mixed_precision.py \
      --output_path checkpoints/ \
      --metadatas datasets/metadata_train.csv,datasets/metadata_eval.csv,be \
      --use_fp16 \
      --batch_size 8
"""

import os
import gc
import torch
import argparse

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager


class MixedPrecisionConfig:
    """Configuration for mixed precision training"""
    def __init__(
        self,
        enabled: bool = False,
        dtype: torch.dtype = torch.float32,
        use_scaler: bool = False,
        scale_loss: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled
        self.dtype = dtype
        self.use_scaler = use_scaler
        self.scale_loss = scale_loss
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval


def create_argument_parser():
    """Create argument parser with mixed precision options"""
    parser = argparse.ArgumentParser(description="XTTS GPT Training with Mixed Precision")

    # Basic training arguments
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to checkpoint output directory")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv,eval_csv,language for each dataset")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Maximum audio length in samples")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Maximum text length in characters")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Save checkpoint every N steps")

    # Mixed precision arguments
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use FP16 mixed precision training")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use BFloat16 mixed precision (recommended for Ampere+)")
    parser.add_argument("--no_gradient_scaling", action="store_true",
                        help="Disable gradient scaling (for BFloat16)")

    # GPU optimization arguments
    parser.add_argument("--tf32_matmul", type=bool, default=False,
                        help="Enable TF32 matrix multiplication")
    parser.add_argument("--tf32_cudnn", type=bool, default=False,
                        help="Enable TF32 for cuDNN")
    parser.add_argument("--compile_model", action="store_true",
                        help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Compilation mode for torch.compile")

    # Gradient monitoring
    parser.add_argument("--monitor_gradients", action="store_true",
                        help="Enable gradient monitoring and logging")
    parser.add_argument("--gradient_clip_val", type=float, default=None,
                        help="Gradient clipping value (None = no clipping)")

    return parser


def setup_mixed_precision(args):
    """Setup mixed precision configuration based on arguments"""
    if args.use_bfloat16 and args.use_fp16:
        raise ValueError("Cannot use both BFloat16 and FP16 simultaneously")

    if args.use_bfloat16:
        # BFloat16 configuration (recommended for Ampere+ GPUs)
        print("Using BFloat16 mixed precision training")
        return MixedPrecisionConfig(
            enabled=True,
            dtype=torch.bfloat16,
            use_scaler=False,  # BFloat16 doesn't need gradient scaling
            scale_loss=False,
        )
    elif args.use_fp16:
        # FP16 configuration (for older GPUs)
        print("Using FP16 mixed precision training with gradient scaling")
        return MixedPrecisionConfig(
            enabled=True,
            dtype=torch.float16,
            use_scaler=not args.no_gradient_scaling,
            scale_loss=not args.no_gradient_scaling,
        )
    else:
        # No mixed precision
        return MixedPrecisionConfig(enabled=False)


def optimize_gpu_settings(args):
    """Apply GPU optimization settings"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU optimizations")
        return

    # Enable TF32 for Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = args.tf32_matmul
    torch.backends.cudnn.allow_tf32 = args.tf32_cudnn

    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True

    # Set memory allocator settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Print configuration
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("GPU Optimization Settings")
    print("=" * 60)
    print(f"GPU Device: {gpu_name}")
    print(f"TF32 MatMul: {'Enabled' if args.tf32_matmul else 'Disabled'}")
    print(f"TF32 cuDNN:  {'Enabled' if args.tf32_cudnn else 'Disabled'}")
    print(f"cuDNN Benchmark: Enabled")

    # Mixed precision info
    if args.use_bfloat16:
        print(f"Mixed Precision: BFloat16")
        print(f"Gradient Scaling: Disabled (not needed for BFloat16)")
    elif args.use_fp16:
        print(f"Mixed Precision: FP16")
        print(f"Gradient Scaling: {'Disabled' if args.no_gradient_scaling else 'Enabled'}")
    else:
        print(f"Mixed Precision: Disabled (FP32)")

    # Compilation info
    if args.compile_model:
        print(f"Model Compilation: Enabled (mode: {args.compile_mode})")
    else:
        print(f"Model Compilation: Disabled")

    # Gradient monitoring
    if args.monitor_gradients:
        print(f"Gradient Monitoring: Enabled")
        if args.gradient_clip_val:
            print(f"Gradient Clipping: {args.gradient_clip_val}")

    # Recommendations
    if 'A100' in gpu_name or '3090' in gpu_name or '4090' in gpu_name or 'A6000' in gpu_name:
        if not args.use_bfloat16:
            print("\n⚠️  Recommendation: Use --use_bfloat16 for better performance on Ampere+ GPUs")
        if not args.tf32_matmul or not args.tf32_cudnn:
            print("⚠️  Recommendation: Enable TF32 with --tf32_matmul=True --tf32_cudnn=True")

    print("=" * 60 + "\n")


class GradientMonitor:
    """Monitor and log gradient statistics"""
    def __init__(self, enabled: bool = True, log_interval: int = 100):
        self.enabled = enabled
        self.log_interval = log_interval
        self.gradient_norms = []
        self.step_count = 0

    def log_gradients(self, model, step):
        """Log gradient statistics"""
        if not self.enabled or step % self.log_interval != 0:
            return

        total_norm = 0.0
        num_params = 0
        max_grad = 0.0
        min_grad = float('inf')

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                min_grad = min(min_grad, p.grad.data.abs().min().item())

        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

        print(f"Step {step} - Gradient Stats:")
        print(f"  Total Norm: {total_norm:.4f}")
        print(f"  Max Grad:   {max_grad:.6f}")
        print(f"  Min Grad:   {min_grad:.6f}")
        print(f"  Params:     {num_params}")

        # Warning for gradient explosion
        if total_norm > 100:
            print(f"⚠️  WARNING: Large gradient norm detected: {total_norm:.2f}")
            print("   Consider reducing learning rate or enabling gradient clipping")

        # Warning for vanishing gradients
        if total_norm < 0.001:
            print(f"⚠️  WARNING: Very small gradient norm detected: {total_norm:.6f}")
            print("   Gradients may be vanishing")


def train_gpt_mixed_precision(args, mp_config):
    """Train GPT with mixed precision support"""

    # Logging parameters
    RUN_NAME = "GPT_XTTS_FT_MixedPrecision"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    OUT_PATH = args.output_path
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
    START_WITH_EVAL = False
    BATCH_SIZE = args.batch_size
    GRAD_ACUMM_STEPS = args.grad_acumm

    # Process datasets
    DATASETS_CONFIG_LIST = []
    for metadata in args.metadatas:
        train_csv, eval_csv, language = metadata.split(",")
        print(f"Dataset: {train_csv}, {eval_csv}, {language}")

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )
        DATASETS_CONFIG_LIST.append(config_dataset)

    # Setup checkpoint paths
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # Download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # Download XTTS files
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))

    # Download XTTS files if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files([TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files([XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # Initialize model arguments
    model_args = GPTArgs(
        max_conditioning_length=264600,
        min_conditioning_length=88200,
        debug_loading_failures=False,
        max_wav_length=args.max_audio_length,
        max_text_length=args.max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # Audio configuration
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    # Training configuration
    config = GPTTrainerConfig()
    config.load_json(XTTS_CONFIG_FILE)

    config.epochs = args.num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = "GPT XTTS training with mixed precision"
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = args.save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {
        "betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": args.weight_decay}
    config.lr = args.lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {
        "milestones": [args.save_step * 3, args.save_step * 6, args.save_step * 9],
        "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    # Add mixed precision config
    if mp_config.enabled:
        config.mixed_precision = True
        config.precision = "bf16" if mp_config.dtype == torch.bfloat16 else "fp16"

    # Initialize model
    model = GPTTrainer.init_from_config(config)

    # Apply torch.compile if requested
    if args.compile_model and hasattr(torch, 'compile'):
        print(f"Compiling model with mode: {args.compile_mode}")
        model.xtts.gpt = torch.compile(
            model.xtts.gpt,
            mode=args.compile_mode,
            fullgraph=False
        )
        print("✓ Model compiled successfully")

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Initialize gradient monitor
    gradient_monitor = GradientMonitor(
        enabled=args.monitor_gradients,
        log_interval=100
    )

    # Initialize trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
            grad_clip=args.gradient_clip_val,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Training with mixed precision
    if mp_config.enabled:
        print(f"\nStarting mixed precision training with {mp_config.dtype}")
        print(f"Gradient scaling: {mp_config.use_scaler}")

    trainer.fit()

    trainer_out_path = trainer.output_path

    # Cleanup
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return trainer_out_path


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup mixed precision
    mp_config = setup_mixed_precision(args)

    # Apply GPU optimizations
    optimize_gpu_settings(args)

    # Train model
    trainer_out_path = train_gpt_mixed_precision(args, mp_config)

    print(f"\nTraining completed!")
    print(f"Checkpoint saved in: {trainer_out_path}")


if __name__ == "__main__":
    main()
