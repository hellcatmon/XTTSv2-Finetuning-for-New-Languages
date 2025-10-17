import os
import gc
import torch

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

import argparse


def create_xtts_trainer_parser():
    parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to pretrained + checkpoint model")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv_path,eval_csv_path,language")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Grad accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Max audio length")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Max text length")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Save step")
    parser.add_argument("--tf32_matmul", type=bool, default=False,
                        help="Enable or disable Torch TF32 MatMul")
    parser.add_argument("--tf32_cudnn", type=bool, default=False,
                        help="Enable or disable Torch TF32 CUDNN")

    return parser


def train_gpt(metadatas, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, max_text_length, lr, weight_decay, save_step):
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    # OUT_PATH = os.path.join(output_path, "run", "training")
    OUT_PATH = output_path

    # Training Parameters
    # for multi-gpu training please make it False
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
    START_WITH_EVAL = False  # if True it will star with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps

    # Define here the dataset that you want to use for the fine-tuning on.
    DATASETS_CONFIG_LIST = []
    for metadata in metadatas:
        train_csv, eval_csv, language = metadata.split(",")
        print(train_csv, eval_csv, language)

        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )

        DATASETS_CONFIG_LIST.append(config_dataset)

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(
        OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(
        CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(
        CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(
        TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(
        XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(
        # config.json file
        CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files(
            [XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files(
            [XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=264600,  # 12 secs
        min_conditioning_length=88200,  # 4 secs
        debug_loading_failures=False,
        max_wav_length=max_audio_length,  # ~11.6 seconds
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        # checkpoint path of the model that you want to fine-tune
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config

    config = GPTTrainerConfig()

    config.load_json(XTTS_CONFIG_FILE)

    # Remove incompatible parameters for newer trainer versions
    # These parameters exist in old config files but are not supported by TrainerArgs anymore
    incompatible_params = ['grad_clip', 'grad_clip_norm_type']
    for param in incompatible_params:
        if hasattr(config, param):
            print(f" > Removing incompatible config parameter: {param}")
            delattr(config, param)
        # Also remove from __dict__ if present (Coqpit stores values there)
        if param in config.__dict__:
            del config.__dict__[param]

    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = """
        GPT XTTS training
        """,
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    # Optimized DataLoader settings for better performance
    config.num_loader_workers = 8  # Increased from 4 for better CPU utilization
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {
        "betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [
        save_step * 3, save_step * 3 * 2, save_step * 3 * 3], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS
        ),
        config,
        # output_path=os.path.join(output_path, "run", "training"),
        output_path=os.path.join(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return trainer_out_path


def optimize_gpu_settings(args):
    """Apply GPU optimization settings"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU optimizations")
        return

    # Enable TF32 for Ampere+ GPUs (3090, 4090, A100, etc.)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32_matmul
    torch.backends.cudnn.allow_tf32 = args.tf32_cudnn

    # Enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Set memory allocator settings for better memory management
    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Print GPU information
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("GPU Optimization Settings")
    print("=" * 60)
    print(f"GPU Device: {gpu_name}")
    print(f"TF32 MatMul: {'Enabled' if args.tf32_matmul else 'Disabled'}")
    print(f"TF32 cuDNN:  {'Enabled' if args.tf32_cudnn else 'Disabled'}")
    print(f"cuDNN Benchmark: Enabled")
    print(f"CUDA Memory Config: max_split_size_mb=512")

    # Recommendation for Ampere+ GPUs
    if 'A100' in gpu_name or '3090' in gpu_name or '4090' in gpu_name or 'A6000' in gpu_name:
        if not args.tf32_matmul or not args.tf32_cudnn:
            print("\n⚠️  Recommendation: Enable TF32 for better performance on Ampere+ GPUs")
            print("   Add: --tf32_matmul=True --tf32_cudnn=True")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = create_xtts_trainer_parser()
    args = parser.parse_args()

    # Apply GPU optimizations
    optimize_gpu_settings(args)

    trainer_out_path = train_gpt(
        metadatas=args.metadatas,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_acumm,
        weight_decay=args.weight_decay,
        lr=args.lr,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        save_step=args.save_step
    )

    print(f"Checkpoint saved in dir: {trainer_out_path}")
