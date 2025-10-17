#!/usr/bin/env python3
"""
Advanced GPT XTTS Training with All P2 Optimizations
Includes: Mixed Precision, Data Augmentation, Gradient Monitoring, torch.compile

Usage:
    # Full optimizations (Ampere+ GPU)
    CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts_advanced.py \
      --output_path checkpoints/ \
      --metadatas datasets/metadata_train.csv,datasets/metadata_eval.csv,be \
      --use_bfloat16 \
      --use_augmentation \
      --augmentation_preset medium \
      --monitor_gradients \
      --compile_model \
      --tf32_matmul=True \
      --tf32_cudnn=True \
      --batch_size 8 \
      --grad_acumm 4

    # Conservative (older GPUs)
    CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts_advanced.py \
      --output_path checkpoints/ \
      --metadatas datasets/metadata_train.csv,datasets/metadata_eval.csv,be \
      --use_fp16 \
      --use_augmentation \
      --batch_size 4 \
      --grad_acumm 8
"""

import os
import sys
import gc
import torch
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Import P2 modules
try:
    from audio_augmentation import get_augmentor, PRESET_CONFIGS
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: audio_augmentation.py not found. Augmentation disabled.")
    AUGMENTATION_AVAILABLE = False

try:
    from gradient_monitor import GradientMonitor
    GRADIENT_MONITOR_AVAILABLE = True
except ImportError:
    print("Warning: gradient_monitor.py not found. Gradient monitoring disabled.")
    GRADIENT_MONITOR_AVAILABLE = False


def create_parser():
    """Create comprehensive argument parser"""
    parser = argparse.ArgumentParser(
        description="Advanced XTTS GPT Training with P1+P2 Optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Basic Training Arguments ===
    basic = parser.add_argument_group('Basic Training')
    basic.add_argument("--output_path", type=str, required=True,
                      help="Path to checkpoint output directory")
    basic.add_argument("--metadatas", nargs='+', type=str, required=True,
                      help="train_csv,eval_csv,language for each dataset")
    basic.add_argument("--num_epochs", type=int, default=8,
                      help="Number of training epochs")
    basic.add_argument("--batch_size", type=int, default=8,
                      help="Training batch size")
    basic.add_argument("--grad_acumm", type=int, default=4,
                      help="Gradient accumulation steps")
    basic.add_argument("--max_audio_length", type=int, default=330750,
                      help="Maximum audio length in samples (~15 seconds)")
    basic.add_argument("--max_text_length", type=int, default=400,
                      help="Maximum text length in characters")
    basic.add_argument("--weight_decay", type=float, default=1e-2,
                      help="Weight decay for optimizer")
    basic.add_argument("--lr", type=float, default=5e-6,
                      help="Learning rate")
    basic.add_argument("--save_step", type=int, default=10000,
                      help="Save checkpoint every N steps")

    # === P1: GPU Optimizations ===
    p1 = parser.add_argument_group('P1: GPU Optimizations')
    p1.add_argument("--tf32_matmul", action="store_true",
                   help="Enable TF32 matrix multiplication (Ampere+ GPUs)")
    p1.add_argument("--tf32_cudnn", action="store_true",
                   help="Enable TF32 for cuDNN (Ampere+ GPUs)")
    p1.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader worker threads")

    # === P2: Mixed Precision ===
    mp = parser.add_argument_group('P2: Mixed Precision Training')
    mp.add_argument("--use_fp16", action="store_true",
                   help="Use FP16 mixed precision (older GPUs)")
    mp.add_argument("--use_bfloat16", action="store_true",
                   help="Use BFloat16 mixed precision (Ampere+ recommended)")
    mp.add_argument("--no_gradient_scaling", action="store_true",
                   help="Disable gradient scaling (for BFloat16)")

    # === P2: Data Augmentation ===
    aug = parser.add_argument_group('P2: Data Augmentation')
    aug.add_argument("--use_augmentation", action="store_true",
                    help="Enable audio data augmentation")
    aug.add_argument("--augmentation_preset", type=str, default="medium",
                    choices=list(PRESET_CONFIGS.keys()) if AUGMENTATION_AVAILABLE else [],
                    help="Augmentation preset (light/medium/heavy)")
    aug.add_argument("--augment_prob", type=float, default=0.3,
                    help="Probability of applying augmentation")

    # === P2: Gradient Monitoring ===
    grad = parser.add_argument_group('P2: Gradient Monitoring')
    grad.add_argument("--monitor_gradients", action="store_true",
                     help="Enable gradient monitoring and logging")
    grad.add_argument("--gradient_clip_val", type=float, default=1.0,
                     help="Gradient clipping value (None = no clipping)")
    grad.add_argument("--gradient_log_interval", type=int, default=100,
                     help="Log gradients every N steps")

    # === P2: Model Compilation ===
    comp = parser.add_argument_group('P2: Model Compilation (PyTorch 2.0+)')
    comp.add_argument("--compile_model", action="store_true",
                     help="Use torch.compile for faster training")
    comp.add_argument("--compile_mode", type=str, default="reduce-overhead",
                     choices=["default", "reduce-overhead", "max-autotune"],
                     help="Compilation mode")

    # === Advanced Options ===
    adv = parser.add_argument_group('Advanced Options')
    adv.add_argument("--profile", action="store_true",
                    help="Enable PyTorch profiler")
    adv.add_argument("--detect_anomaly", action="store_true",
                    help="Enable anomaly detection (slower, for debugging)")

    return parser


def print_configuration(args):
    """Print training configuration"""
    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING CONFIGURATION")
    print("=" * 70)

    print("\n📊 Basic Settings:")
    print(f"  Output Path:        {args.output_path}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Batch Size:         {args.batch_size}")
    print(f"  Grad Accumulation:  {args.grad_acumm}")
    print(f"  Effective Batch:    {args.batch_size * args.grad_acumm}")
    print(f"  Learning Rate:      {args.lr:.2e}")
    print(f"  Weight Decay:       {args.weight_decay}")

    print("\n🎮 GPU Settings (P1):")
    if torch.cuda.is_available():
        print(f"  Device:             {torch.cuda.get_device_name(0)}")
        print(f"  TF32 MatMul:        {'✓ Enabled' if args.tf32_matmul else '✗ Disabled'}")
        print(f"  TF32 cuDNN:         {'✓ Enabled' if args.tf32_cudnn else '✗ Disabled'}")
        print(f"  DataLoader Workers: {args.num_workers}")
    else:
        print(f"  Device:             CPU (CUDA not available)")

    print("\n🔬 Mixed Precision (P2):")
    if args.use_bfloat16:
        print(f"  Type:               BFloat16 ✓")
        print(f"  Gradient Scaling:   Disabled (not needed)")
    elif args.use_fp16:
        print(f"  Type:               FP16 ✓")
        print(f"  Gradient Scaling:   {'Disabled' if args.no_gradient_scaling else 'Enabled ✓'}")
    else:
        print(f"  Type:               FP32 (disabled)")

    print("\n🎨 Data Augmentation (P2):")
    if args.use_augmentation and AUGMENTATION_AVAILABLE:
        print(f"  Status:             ✓ Enabled")
        print(f"  Preset:             {args.augmentation_preset}")
        print(f"  Probability:        {args.augment_prob}")
    else:
        print(f"  Status:             ✗ Disabled")

    print("\n📈 Gradient Monitoring (P2):")
    if args.monitor_gradients and GRADIENT_MONITOR_AVAILABLE:
        print(f"  Status:             ✓ Enabled")
        print(f"  Gradient Clipping:  {args.gradient_clip_val if args.gradient_clip_val else 'Disabled'}")
        print(f"  Log Interval:       Every {args.gradient_log_interval} steps")
    else:
        print(f"  Status:             ✗ Disabled")

    print("\n⚡ Model Compilation (P2):")
    if args.compile_model and hasattr(torch, 'compile'):
        print(f"  Status:             ✓ Enabled")
        print(f"  Mode:               {args.compile_mode}")
    else:
        print(f"  Status:             ✗ Disabled")

    print("\n" + "=" * 70 + "\n")


def optimize_gpu_settings(args):
    """Apply GPU optimizations (P1)"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training on CPU will be very slow.")
        return

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32_matmul
    torch.backends.cudnn.allow_tf32 = args.tf32_cudnn

    # cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Memory allocator
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Anomaly detection
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("⚠️  Anomaly detection enabled (slower, for debugging)")


def train_with_advanced_features(args):
    """Main training function with all P2 features"""

    # === Setup ===
    optimize_gpu_settings(args)

    RUN_NAME = "GPT_XTTS_Advanced"
    OUT_PATH = args.output_path

    # Process datasets
    DATASETS_CONFIG_LIST = []
    for metadata in args.metadatas:
        train_csv, eval_csv, language = metadata.split(",")
        print(f"📁 Dataset: {language} - {train_csv}")

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )
        DATASETS_CONFIG_LIST.append(config_dataset)

    # === Download checkpoints ===
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")

    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print("📥 Downloading DVAE files...")
        ModelManager._download_model_files(
            ["https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth",
             "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"],
            CHECKPOINTS_OUT_PATH, progress_bar=True)

    # XTTS files
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "config.json")

    if not all(os.path.isfile(f) for f in [TOKENIZER_FILE, XTTS_CHECKPOINT, XTTS_CONFIG_FILE]):
        print("📥 Downloading XTTS v2.0 files...")
        ModelManager._download_model_files(
            ["https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
             "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth",
             "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"],
            CHECKPOINTS_OUT_PATH, progress_bar=True)

    # === Model configuration ===
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

    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig()
    config.load_json(XTTS_CONFIG_FILE)

    config.epochs = args.num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = "XTTS_Advanced"
    config.run_description = "Advanced training with P1+P2 optimizations"
    config.dashboard_logger = "tensorboard"
    config.audio = audio_config
    config.batch_size = args.batch_size
    config.num_loader_workers = args.num_workers
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = args.save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = True
    config.optimizer_params = {
        "betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": args.weight_decay}
    config.lr = args.lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {
        "milestones": [args.save_step * 3, args.save_step * 6, args.save_step * 9],
        "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    # === Initialize model ===
    print("\n🔧 Initializing model...")
    model = GPTTrainer.init_from_config(config)

    # === P2: Apply torch.compile ===
    if args.compile_model and hasattr(torch, 'compile'):
        print(f"⚡ Compiling model (mode: {args.compile_mode})...")
        try:
            model.xtts.gpt = torch.compile(
                model.xtts.gpt,
                mode=args.compile_mode,
                fullgraph=False
            )
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")
            print("   Continuing without compilation")

    # === Load training samples ===
    print("\n📚 Loading training samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Eval samples:  {len(eval_samples)}")

    # === P2: Setup augmentation ===
    augmentor = None
    if args.use_augmentation and AUGMENTATION_AVAILABLE:
        print(f"\n🎨 Setting up data augmentation ({args.augmentation_preset} preset)...")
        augmentor = get_augmentor(
            preset=args.augmentation_preset,
            sample_rate=22050,
            augment_prob=args.augment_prob
        )
        print("✓ Augmentation enabled")

    # === P2: Setup gradient monitoring ===
    gradient_monitor = None
    if args.monitor_gradients and GRADIENT_MONITOR_AVAILABLE:
        print(f"\n📈 Setting up gradient monitoring...")
        gradient_monitor = GradientMonitor(
            log_dir=os.path.join(OUT_PATH, "gradient_logs"),
            log_interval=args.gradient_log_interval,
            enable_tensorboard=True,
        )
        print("✓ Gradient monitoring enabled")

    # === Initialize trainer ===
    print("\n🚀 Initializing trainer...")
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=args.grad_acumm,
            grad_clip=args.gradient_clip_val if args.gradient_clip_val else None,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # === Start training ===
    print("\n" + "=" * 70)
    print(" " * 25 + "STARTING TRAINING")
    print("=" * 70 + "\n")

    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if gradient_monitor is not None:
            print("\n📊 Saving gradient statistics...")
            gradient_monitor.close()

    trainer_out_path = trainer.output_path

    # Final cleanup
    del model, trainer, train_samples, eval_samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trainer_out_path


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.use_bfloat16 and args.use_fp16:
        print("❌ Error: Cannot use both BFloat16 and FP16 simultaneously")
        sys.exit(1)

    if args.use_augmentation and not AUGMENTATION_AVAILABLE:
        print("⚠️  Warning: Augmentation requested but module not available")
        args.use_augmentation = False

    if args.monitor_gradients and not GRADIENT_MONITOR_AVAILABLE:
        print("⚠️  Warning: Gradient monitoring requested but module not available")
        args.monitor_gradients = False

    # Print configuration
    print_configuration(args)

    # Train
    try:
        trainer_out_path = train_with_advanced_features(args)

        print("\n" + "=" * 70)
        print(" " * 25 + "TRAINING COMPLETED!")
        print("=" * 70)
        print(f"\n✓ Checkpoint saved in: {trainer_out_path}")
        print(f"✓ TensorBoard logs: {os.path.join(trainer_out_path, 'tensorboard')}")
        if args.monitor_gradients:
            print(f"✓ Gradient logs: {os.path.join(trainer_out_path, 'gradient_logs')}")
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
