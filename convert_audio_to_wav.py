#!/usr/bin/env python3
"""
Audio Conversion Module for XTTSv2 Finetuning
Converts any audio format (MP3, FLAC, OGG, etc.) to optimal WAV format for training

Usage:
    python convert_audio_to_wav.py --input_dir datasets/raw --output_dir datasets/wav
    python convert_audio_to_wav.py --input_dir datasets/raw --output_dir datasets/wav --sample_rate 22050
    python convert_audio_to_wav.py --input_dir datasets/raw --output_dir datasets/wav --update_metadata metadata.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import csv

import torch
import torchaudio
from tqdm import tqdm


class AudioConverter:
    """Convert audio files to optimal WAV format for XTTS training"""

    def __init__(
        self,
        target_sample_rate: int = 22050,
        target_channels: int = 1,
        target_bit_depth: int = 16,
        normalize: bool = True,
        normalize_peak: float = 0.95,
    ):
        """
        Initialize AudioConverter

        Args:
            target_sample_rate: Target sample rate (default: 22050 Hz for XTTS)
            target_channels: Target number of channels (default: 1 for mono)
            target_bit_depth: Target bit depth (default: 16)
            normalize: Whether to normalize audio (default: True)
            normalize_peak: Peak normalization level (default: 0.95)
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_bit_depth = target_bit_depth
        self.normalize = normalize
        self.normalize_peak = normalize_peak

        # Supported input formats
        self.supported_formats = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

    def convert_to_mono(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono by averaging channels"""
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        return wav

    def normalize_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Normalize audio to prevent clipping"""
        max_val = torch.max(torch.abs(wav))
        if max_val > 0:
            wav = wav / (max_val + 1e-8) * self.normalize_peak
        return wav

    def resample_audio(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        if orig_sr != self.target_sample_rate:
            wav = torchaudio.functional.resample(
                wav, orig_sr, self.target_sample_rate
            )
        return wav

    def convert_file(self, input_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Convert a single audio file to WAV

        Args:
            input_path: Path to input audio file
            output_path: Path to output WAV file

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load audio file
            wav, sr = torchaudio.load(str(input_path))

            # Convert to mono if needed
            if self.target_channels == 1:
                wav = self.convert_to_mono(wav)

            # Resample if needed
            wav = self.resample_audio(wav, sr)

            # Normalize if requested
            if self.normalize:
                wav = self.normalize_audio(wav)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as WAV
            torchaudio.save(
                str(output_path),
                wav,
                self.target_sample_rate,
                bits_per_sample=self.target_bit_depth,
            )

            return True, f"Successfully converted: {input_path.name}"

        except Exception as e:
            return False, f"Error converting {input_path.name}: {str(e)}"

    def batch_convert(
        self, input_dir: Path, output_dir: Path, preserve_structure: bool = True
    ) -> Tuple[int, int, List[str]]:
        """
        Convert all audio files in a directory

        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for WAV files
            preserve_structure: Whether to preserve directory structure

        Returns:
            Tuple of (successful_count, failed_count, error_messages)
        """
        # Find all audio files
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(input_dir.rglob(f'*{ext}'))

        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return 0, 0, []

        print(f"Found {len(audio_files)} audio files to convert")

        successful = 0
        failed = 0
        error_messages = []

        # Convert each file
        for input_path in tqdm(audio_files, desc="Converting audio files"):
            # Determine output path
            if preserve_structure:
                relative_path = input_path.relative_to(input_dir)
                output_path = output_dir / relative_path.with_suffix('.wav')
            else:
                output_path = output_dir / input_path.with_suffix('.wav').name

            # Convert file
            success, message = self.convert_file(input_path, output_path)

            if success:
                successful += 1
            else:
                failed += 1
                error_messages.append(message)

        return successful, failed, error_messages


def update_metadata_paths(
    metadata_path: Path,
    old_audio_dir: Path,
    new_audio_dir: Path,
    output_metadata_path: Optional[Path] = None,
) -> int:
    """
    Update audio file paths in metadata CSV

    Args:
        metadata_path: Path to metadata CSV file
        old_audio_dir: Original audio directory
        new_audio_dir: New audio directory (with WAV files)
        output_metadata_path: Output metadata path (if None, overwrites original)

    Returns:
        Number of updated entries
    """
    if output_metadata_path is None:
        output_metadata_path = metadata_path

    updated_count = 0
    rows = []

    # Read metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) >= 1:
                # Update audio file path (first column)
                old_path = row[0]
                # Convert to WAV extension
                new_path = str(Path(old_path).with_suffix('.wav'))
                # Update directory if needed
                if str(old_audio_dir) in old_path:
                    new_path = new_path.replace(str(old_audio_dir), str(new_audio_dir))

                row[0] = new_path
                updated_count += 1

            rows.append(row)

    # Write updated metadata
    with open(output_metadata_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(rows)

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files to optimal WAV format for XTTS training"
    )

    # Input/Output paths
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing audio files (MP3, FLAC, etc.)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for converted WAV files'
    )

    # Audio conversion settings
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=22050,
        help='Target sample rate in Hz (default: 22050 for XTTS)'
    )
    parser.add_argument(
        '--bit_depth',
        type=int,
        default=16,
        choices=[16, 24, 32],
        help='Target bit depth (default: 16)'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Disable audio normalization'
    )
    parser.add_argument(
        '--normalize_peak',
        type=float,
        default=0.95,
        help='Peak normalization level (default: 0.95)'
    )

    # Directory structure
    parser.add_argument(
        '--flatten',
        action='store_true',
        help='Flatten directory structure (put all files in output root)'
    )

    # Metadata update
    parser.add_argument(
        '--update_metadata',
        type=str,
        help='Path to metadata CSV file to update with new WAV paths'
    )
    parser.add_argument(
        '--output_metadata',
        type=str,
        help='Output path for updated metadata (default: overwrite original)'
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize converter
    converter = AudioConverter(
        target_sample_rate=args.sample_rate,
        target_channels=1,  # Always mono for TTS
        target_bit_depth=args.bit_depth,
        normalize=not args.no_normalize,
        normalize_peak=args.normalize_peak,
    )

    # Print configuration
    print("=" * 60)
    print("Audio Conversion Configuration")
    print("=" * 60)
    print(f"Input directory:    {input_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Target sample rate: {args.sample_rate} Hz")
    print(f"Target bit depth:   {args.bit_depth} bits")
    print(f"Target channels:    1 (mono)")
    print(f"Normalize audio:    {not args.no_normalize}")
    if not args.no_normalize:
        print(f"Normalize peak:     {args.normalize_peak}")
    print(f"Preserve structure: {not args.flatten}")
    print("=" * 60)

    # Convert files
    successful, failed, error_messages = converter.batch_convert(
        input_dir, output_dir, preserve_structure=not args.flatten
    )

    # Print results
    print("\n" + "=" * 60)
    print("Conversion Results")
    print("=" * 60)
    print(f"Successfully converted: {successful} files")
    print(f"Failed conversions:     {failed} files")

    if error_messages:
        print("\nErrors:")
        for msg in error_messages[:10]:  # Show first 10 errors
            print(f"  - {msg}")
        if len(error_messages) > 10:
            print(f"  ... and {len(error_messages) - 10} more errors")

    # Update metadata if requested
    if args.update_metadata:
        metadata_path = Path(args.update_metadata)
        if metadata_path.exists():
            output_metadata_path = (
                Path(args.output_metadata) if args.output_metadata else None
            )
            updated = update_metadata_paths(
                metadata_path, input_dir, output_dir, output_metadata_path
            )
            print(f"\nUpdated {updated} entries in metadata file")
            if output_metadata_path:
                print(f"Updated metadata saved to: {output_metadata_path}")
        else:
            print(f"\nWarning: Metadata file not found: {metadata_path}")

    print("=" * 60)

    # Exit with error code if any conversions failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
