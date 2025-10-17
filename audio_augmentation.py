#!/usr/bin/env python3
"""
Audio Data Augmentation for XTTS Training
Provides various audio augmentation techniques to improve model robustness

Usage:
    from audio_augmentation import AudioAugmentor

    augmentor = AudioAugmentor(sample_rate=22050, augment_prob=0.3)
    augmented_wav = augmentor.apply_augmentation(wav)
"""

import random
import torch
import torchaudio
import torchaudio.functional as F
from typing import Optional, Tuple


class AudioAugmentor:
    """
    Audio augmentation for TTS training

    Applies various augmentations to improve model robustness:
    - Random gain adjustment
    - Gaussian noise addition
    - Time stretching (optional)
    - Pitch shifting (optional)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        augment_prob: float = 0.3,
        gain_range: Tuple[float, float] = (-3.0, 3.0),
        noise_snr_range: Tuple[float, float] = (30.0, 50.0),
        enable_time_stretch: bool = False,
        time_stretch_range: Tuple[float, float] = (0.95, 1.05),
        enable_pitch_shift: bool = False,
        pitch_shift_range: Tuple[int, int] = (-2, 2),
    ):
        """
        Initialize AudioAugmentor

        Args:
            sample_rate: Audio sample rate (default: 22050)
            augment_prob: Probability of applying each augmentation (default: 0.3)
            gain_range: Range of gain adjustment in dB (default: -3 to +3 dB)
            noise_snr_range: Range of SNR for noise addition in dB (default: 30-50 dB)
            enable_time_stretch: Enable time stretching augmentation (default: False)
            time_stretch_range: Range of time stretch factors (default: 0.95-1.05)
            enable_pitch_shift: Enable pitch shifting augmentation (default: False)
            pitch_shift_range: Range of pitch shift in semitones (default: -2 to +2)
        """
        self.sample_rate = sample_rate
        self.augment_prob = augment_prob
        self.gain_range = gain_range
        self.noise_snr_range = noise_snr_range
        self.enable_time_stretch = enable_time_stretch
        self.time_stretch_range = time_stretch_range
        self.enable_pitch_shift = enable_pitch_shift
        self.pitch_shift_range = pitch_shift_range

    def apply_random_gain(
        self,
        wav: torch.Tensor,
        gain_range: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Apply random gain adjustment in dB

        Args:
            wav: Input audio tensor
            gain_range: Range of gain in dB (default: use instance setting)

        Returns:
            Audio with random gain applied
        """
        if gain_range is None:
            gain_range = self.gain_range

        gain_db = random.uniform(gain_range[0], gain_range[1])
        gain_linear = 10 ** (gain_db / 20)
        return wav * gain_linear

    def add_gaussian_noise(
        self,
        wav: torch.Tensor,
        snr_range: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Add Gaussian noise at specified SNR

        Args:
            wav: Input audio tensor
            snr_range: Range of SNR in dB (default: use instance setting)

        Returns:
            Audio with noise added
        """
        if snr_range is None:
            snr_range = self.noise_snr_range

        snr_db = random.uniform(snr_range[0], snr_range[1])

        # Calculate signal power
        signal_power = torch.mean(wav ** 2)

        # Calculate noise power from SNR
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate and add noise
        noise = torch.randn_like(wav) * torch.sqrt(noise_power)
        return wav + noise

    def time_stretch(
        self,
        wav: torch.Tensor,
        stretch_range: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Apply time stretching (speed perturbation)

        Args:
            wav: Input audio tensor
            stretch_range: Range of stretch factors (default: use instance setting)

        Returns:
            Time-stretched audio
        """
        if stretch_range is None:
            stretch_range = self.time_stretch_range

        stretch_factor = random.uniform(stretch_range[0], stretch_range[1])

        # Resample to simulate time stretching
        new_sr = int(self.sample_rate * stretch_factor)
        stretched = F.resample(wav, self.sample_rate, new_sr)

        # Resample back to original sample rate
        wav_stretched = F.resample(stretched, new_sr, self.sample_rate)

        return wav_stretched

    def pitch_shift(
        self,
        wav: torch.Tensor,
        shift_range: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Apply pitch shifting

        Args:
            wav: Input audio tensor
            shift_range: Range of pitch shift in semitones (default: use instance setting)

        Returns:
            Pitch-shifted audio
        """
        if shift_range is None:
            shift_range = self.pitch_shift_range

        n_steps = random.randint(shift_range[0], shift_range[1])

        if n_steps == 0:
            return wav

        # Calculate pitch shift factor
        factor = 2 ** (n_steps / 12)

        # Resample to shift pitch
        new_sr = int(self.sample_rate * factor)
        shifted = F.resample(wav, self.sample_rate, new_sr)

        # Resample back to maintain original length
        wav_shifted = F.resample(shifted, new_sr, self.sample_rate)

        return wav_shifted

    def apply_augmentation(
        self,
        wav: torch.Tensor,
        force_augment: bool = False
    ) -> torch.Tensor:
        """
        Apply random augmentations to audio

        Args:
            wav: Input audio tensor (shape: [channels, samples])
            force_augment: Force at least one augmentation (default: False)

        Returns:
            Augmented audio tensor
        """
        original_device = wav.device
        wav = wav.cpu()  # Move to CPU for augmentation

        augmented = False

        # Apply gain augmentation
        if random.random() < self.augment_prob or (force_augment and not augmented):
            wav = self.apply_random_gain(wav)
            augmented = True

        # Apply noise augmentation (with lower probability)
        if random.random() < self.augment_prob * 0.5 or (force_augment and not augmented):
            wav = self.add_gaussian_noise(wav)
            augmented = True

        # Apply time stretch (if enabled)
        if self.enable_time_stretch and (random.random() < self.augment_prob * 0.3):
            wav = self.time_stretch(wav)
            augmented = True

        # Apply pitch shift (if enabled)
        if self.enable_pitch_shift and (random.random() < self.augment_prob * 0.2):
            wav = self.pitch_shift(wav)
            augmented = True

        # Clip to prevent overflow
        wav = wav.clip(-1.0, 1.0)

        return wav.to(original_device)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Convenience method for augmentation"""
        return self.apply_augmentation(wav)


class SpecAugmentor:
    """
    Spectrogram-based augmentation (SpecAugment)
    Can be applied to mel spectrograms
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 70,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        augment_prob: float = 0.3,
    ):
        """
        Initialize SpecAugmentor

        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            augment_prob: Probability of applying augmentation
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.augment_prob = augment_prob

        self.spec_augment = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_augment = torchaudio.transforms.TimeMasking(time_mask_param)

    def apply_augmentation(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram

        Args:
            spec: Spectrogram tensor (shape: [channels, freq_bins, time_frames])

        Returns:
            Augmented spectrogram
        """
        if random.random() > self.augment_prob:
            return spec

        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            spec = self.spec_augment(spec)

        # Apply time masking
        for _ in range(self.num_time_masks):
            spec = self.time_augment(spec)

        return spec

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Convenience method for augmentation"""
        return self.apply_augmentation(spec)


class AugmentationPipeline:
    """
    Complete augmentation pipeline for training
    Combines audio and spectrogram augmentations
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        use_audio_aug: bool = True,
        use_spec_aug: bool = False,
        audio_aug_prob: float = 0.3,
        spec_aug_prob: float = 0.3,
        **kwargs
    ):
        """
        Initialize augmentation pipeline

        Args:
            sample_rate: Audio sample rate
            use_audio_aug: Enable audio augmentation
            use_spec_aug: Enable spectrogram augmentation
            audio_aug_prob: Probability for audio augmentation
            spec_aug_prob: Probability for spectrogram augmentation
            **kwargs: Additional arguments for augmentors
        """
        self.use_audio_aug = use_audio_aug
        self.use_spec_aug = use_spec_aug

        if use_audio_aug:
            self.audio_augmentor = AudioAugmentor(
                sample_rate=sample_rate,
                augment_prob=audio_aug_prob,
                **{k: v for k, v in kwargs.items() if k in AudioAugmentor.__init__.__code__.co_varnames}
            )

        if use_spec_aug:
            self.spec_augmentor = SpecAugmentor(
                augment_prob=spec_aug_prob,
                **{k: v for k, v in kwargs.items() if k in SpecAugmentor.__init__.__code__.co_varnames}
            )

    def augment_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation"""
        if self.use_audio_aug:
            return self.audio_augmentor(wav)
        return wav

    def augment_spec(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram augmentation"""
        if self.use_spec_aug:
            return self.spec_augmentor(spec)
        return spec

    def __call__(self, wav: torch.Tensor, spec: Optional[torch.Tensor] = None):
        """
        Apply full augmentation pipeline

        Args:
            wav: Audio waveform
            spec: Spectrogram (optional)

        Returns:
            Augmented audio and spectrogram (if provided)
        """
        aug_wav = self.augment_audio(wav)

        if spec is not None:
            aug_spec = self.augment_spec(spec)
            return aug_wav, aug_spec

        return aug_wav


# Preset configurations for different use cases
PRESET_CONFIGS = {
    "light": {
        "augment_prob": 0.2,
        "gain_range": (-2.0, 2.0),
        "noise_snr_range": (40.0, 60.0),
        "enable_time_stretch": False,
        "enable_pitch_shift": False,
    },
    "medium": {
        "augment_prob": 0.3,
        "gain_range": (-3.0, 3.0),
        "noise_snr_range": (30.0, 50.0),
        "enable_time_stretch": True,
        "time_stretch_range": (0.97, 1.03),
        "enable_pitch_shift": False,
    },
    "heavy": {
        "augment_prob": 0.5,
        "gain_range": (-4.0, 4.0),
        "noise_snr_range": (25.0, 45.0),
        "enable_time_stretch": True,
        "time_stretch_range": (0.95, 1.05),
        "enable_pitch_shift": True,
        "pitch_shift_range": (-2, 2),
    },
}


def get_augmentor(preset: str = "medium", sample_rate: int = 22050, **kwargs) -> AudioAugmentor:
    """
    Get an audio augmentor with preset configuration

    Args:
        preset: Preset name ("light", "medium", "heavy")
        sample_rate: Audio sample rate
        **kwargs: Override preset parameters

    Returns:
        Configured AudioAugmentor
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESET_CONFIGS.keys())}")

    config = PRESET_CONFIGS[preset].copy()
    config.update(kwargs)
    config["sample_rate"] = sample_rate

    return AudioAugmentor(**config)


# Example usage and testing
if __name__ == "__main__":
    print("Audio Augmentation Module")
    print("=" * 60)

    # Create test audio (1 second of sine wave)
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0  # A4 note

    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_wav = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0) * 0.5

    print(f"Test audio shape: {test_wav.shape}")
    print(f"Test audio range: [{test_wav.min():.3f}, {test_wav.max():.3f}]")

    # Test different presets
    for preset_name in ["light", "medium", "heavy"]:
        print(f"\nTesting '{preset_name}' preset:")
        augmentor = get_augmentor(preset_name, sample_rate)

        augmented = augmentor(test_wav.clone())
        print(f"  Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
        print(f"  Energy change: {(augmented.pow(2).mean() / test_wav.pow(2).mean()).item():.2f}x")

    # Test pipeline
    print("\nTesting augmentation pipeline:")
    pipeline = AugmentationPipeline(
        sample_rate=sample_rate,
        use_audio_aug=True,
        use_spec_aug=False,
        audio_aug_prob=1.0,  # Force augmentation for testing
    )

    augmented = pipeline(test_wav.clone())
    print(f"  Pipeline output range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    print("\n" + "=" * 60)
    print("Augmentation module loaded successfully!")
    print("Use: from audio_augmentation import get_augmentor")
