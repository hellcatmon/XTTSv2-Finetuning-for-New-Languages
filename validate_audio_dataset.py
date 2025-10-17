#!/usr/bin/env python3
"""
Audio Dataset Validation Module for XTTS v2 Finetuning
Validates audio quality and generates detailed reports

Usage:
    python validate_audio_dataset.py --metadata datasets/metadata_train.csv
    python validate_audio_dataset.py --metadata datasets/metadata_train.csv --output_report validation_report.json
    python validate_audio_dataset.py --metadata datasets/metadata_train.csv --fix_issues --output_fixed metadata_clean.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
from tqdm import tqdm


class AudioValidator:
    """Validate audio files for XTTS training"""

    def __init__(
        self,
        target_sample_rate: int = 22050,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        clipping_threshold: float = 0.99,
        silence_threshold: float = 0.01,
        max_silence_ratio: float = 0.5,
        min_rms_energy: float = 0.01,
        max_dc_offset: float = 0.1,
    ):
        """
        Initialize AudioValidator

        Args:
            target_sample_rate: Expected sample rate (default: 22050 Hz)
            min_duration: Minimum audio duration in seconds (default: 1.0)
            max_duration: Maximum audio duration in seconds (default: 15.0)
            clipping_threshold: Threshold for detecting clipping (default: 0.99)
            silence_threshold: Threshold for detecting silence (default: 0.01)
            max_silence_ratio: Maximum allowed silence ratio (default: 0.5)
            min_rms_energy: Minimum RMS energy (default: 0.01)
            max_dc_offset: Maximum DC offset (default: 0.1)
        """
        self.target_sample_rate = target_sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.clipping_threshold = clipping_threshold
        self.silence_threshold = silence_threshold
        self.max_silence_ratio = max_silence_ratio
        self.min_rms_energy = min_rms_energy
        self.max_dc_offset = max_dc_offset

    def check_clipping(self, wav: torch.Tensor) -> Tuple[bool, float]:
        """
        Check for audio clipping

        Args:
            wav: Audio waveform tensor

        Returns:
            Tuple of (has_clipping, max_amplitude)
        """
        max_amp = torch.max(torch.abs(wav)).item()
        has_clipping = max_amp >= self.clipping_threshold
        return has_clipping, max_amp

    def check_silence(self, wav: torch.Tensor) -> Tuple[bool, float]:
        """
        Check for excessive silence

        Args:
            wav: Audio waveform tensor

        Returns:
            Tuple of (has_excessive_silence, silence_ratio)
        """
        silence_mask = torch.abs(wav) < self.silence_threshold
        silence_ratio = silence_mask.float().mean().item()
        has_excessive_silence = silence_ratio > self.max_silence_ratio
        return has_excessive_silence, silence_ratio

    def check_energy(self, wav: torch.Tensor) -> Tuple[bool, float]:
        """
        Check RMS energy level

        Args:
            wav: Audio waveform tensor

        Returns:
            Tuple of (has_low_energy, rms_energy)
        """
        rms = torch.sqrt(torch.mean(wav ** 2)).item()
        has_low_energy = rms < self.min_rms_energy
        return has_low_energy, rms

    def check_dc_offset(self, wav: torch.Tensor) -> Tuple[bool, float]:
        """
        Check for DC offset

        Args:
            wav: Audio waveform tensor

        Returns:
            Tuple of (has_dc_offset, offset_value)
        """
        dc_offset = torch.mean(wav).item()
        has_dc_offset = abs(dc_offset) > self.max_dc_offset
        return has_dc_offset, dc_offset

    def check_duration(
        self, wav: torch.Tensor, sample_rate: int
    ) -> Tuple[bool, float]:
        """
        Check audio duration

        Args:
            wav: Audio waveform tensor
            sample_rate: Sample rate

        Returns:
            Tuple of (is_valid_duration, duration_seconds)
        """
        duration = wav.shape[-1] / sample_rate
        is_valid = self.min_duration <= duration <= self.max_duration
        return is_valid, duration

    def check_sample_rate(self, sample_rate: int) -> Tuple[bool, int]:
        """
        Check sample rate

        Args:
            sample_rate: Audio sample rate

        Returns:
            Tuple of (is_correct_rate, sample_rate)
        """
        is_correct = sample_rate == self.target_sample_rate
        return is_correct, sample_rate

    def validate_audio_file(
        self, audio_path: str
    ) -> Tuple[bool, Dict[str, any], List[str]]:
        """
        Validate a single audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (is_valid, metrics, issues)
        """
        issues = []
        metrics = {}

        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)

            # Convert to mono if needed
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Check sample rate
            correct_sr, current_sr = self.check_sample_rate(sr)
            metrics['sample_rate'] = current_sr
            if not correct_sr:
                issues.append(f"wrong_sample_rate_{current_sr}")

            # Check duration
            valid_duration, duration = self.check_duration(wav, sr)
            metrics['duration'] = duration
            if not valid_duration:
                if duration < self.min_duration:
                    issues.append(f"too_short_{duration:.2f}s")
                else:
                    issues.append(f"too_long_{duration:.2f}s")

            # Check clipping
            has_clipping, max_amp = self.check_clipping(wav)
            metrics['max_amplitude'] = max_amp
            if has_clipping:
                issues.append(f"clipping_{max_amp:.3f}")

            # Check silence
            has_excessive_silence, silence_ratio = self.check_silence(wav)
            metrics['silence_ratio'] = silence_ratio
            if has_excessive_silence:
                issues.append(f"excessive_silence_{silence_ratio:.2f}")

            # Check energy
            has_low_energy, rms = self.check_energy(wav)
            metrics['rms_energy'] = rms
            if has_low_energy:
                issues.append(f"low_energy_{rms:.4f}")

            # Check DC offset
            has_dc_offset, dc_value = self.check_dc_offset(wav)
            metrics['dc_offset'] = dc_value
            if has_dc_offset:
                issues.append(f"dc_offset_{dc_value:.3f}")

            # Determine if valid
            is_valid = len(issues) == 0

            return is_valid, metrics, issues

        except Exception as e:
            issues.append(f"load_error: {str(e)}")
            return False, {}, issues

    def validate_dataset(
        self, metadata_path: str, base_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Validate entire dataset

        Args:
            metadata_path: Path to metadata CSV file
            base_path: Base path for audio files (if not absolute in metadata)

        Returns:
            Validation report dictionary
        """
        print("=" * 60)
        print("Audio Dataset Validation")
        print("=" * 60)

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            rows = list(reader)

        print(f"Total samples in metadata: {len(rows)}")

        # Validation results
        results = {
            'total_samples': len(rows),
            'valid_samples': 0,
            'invalid_samples': 0,
            'issues_summary': {},
            'valid_files': [],
            'invalid_files': [],
            'metrics_summary': {
                'duration': {'min': float('inf'), 'max': 0, 'avg': 0},
                'rms_energy': {'min': float('inf'), 'max': 0, 'avg': 0},
                'max_amplitude': {'min': float('inf'), 'max': 0, 'avg': 0},
            }
        }

        # Validate each file
        durations = []
        rms_energies = []
        max_amplitudes = []

        for row in tqdm(rows, desc="Validating audio files"):
            if len(row) < 1:
                continue

            audio_file = row[0]

            # Resolve full path
            if base_path and not os.path.isabs(audio_file):
                full_path = os.path.join(base_path, audio_file)
            else:
                full_path = audio_file

            # Validate file
            is_valid, metrics, issues = self.validate_audio_file(full_path)

            if is_valid:
                results['valid_samples'] += 1
                results['valid_files'].append(audio_file)

                # Collect metrics
                if 'duration' in metrics:
                    durations.append(metrics['duration'])
                if 'rms_energy' in metrics:
                    rms_energies.append(metrics['rms_energy'])
                if 'max_amplitude' in metrics:
                    max_amplitudes.append(metrics['max_amplitude'])
            else:
                results['invalid_samples'] += 1
                results['invalid_files'].append({
                    'file': audio_file,
                    'issues': issues
                })

                # Count issues
                for issue in issues:
                    issue_type = issue.split('_')[0]
                    if issue_type not in results['issues_summary']:
                        results['issues_summary'][issue_type] = 0
                    results['issues_summary'][issue_type] += 1

        # Calculate summary metrics
        if durations:
            results['metrics_summary']['duration'] = {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations)
            }

        if rms_energies:
            results['metrics_summary']['rms_energy'] = {
                'min': min(rms_energies),
                'max': max(rms_energies),
                'avg': sum(rms_energies) / len(rms_energies)
            }

        if max_amplitudes:
            results['metrics_summary']['max_amplitude'] = {
                'min': min(max_amplitudes),
                'max': max(max_amplitudes),
                'avg': sum(max_amplitudes) / len(max_amplitudes)
            }

        return results

    def print_report(self, results: Dict[str, any]) -> None:
        """
        Print validation report

        Args:
            results: Validation results dictionary
        """
        print("\n" + "=" * 60)
        print("Validation Report")
        print("=" * 60)

        print(f"\nOverall Statistics:")
        print(f"  Total samples:   {results['total_samples']}")
        print(f"  Valid samples:   {results['valid_samples']} ({results['valid_samples']/results['total_samples']*100:.1f}%)")
        print(f"  Invalid samples: {results['invalid_samples']} ({results['invalid_samples']/results['total_samples']*100:.1f}%)")

        if results['issues_summary']:
            print(f"\nIssues Summary:")
            for issue_type, count in sorted(results['issues_summary'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {issue_type:20s}: {count} files")

        print(f"\nMetrics Summary (Valid Files Only):")
        for metric_name, stats in results['metrics_summary'].items():
            if stats['max'] > 0:
                print(f"  {metric_name}:")
                print(f"    Min: {stats['min']:.4f}")
                print(f"    Max: {stats['max']:.4f}")
                print(f"    Avg: {stats['avg']:.4f}")

        if results['invalid_samples'] > 0 and results['invalid_samples'] <= 10:
            print(f"\nInvalid Files:")
            for item in results['invalid_files']:
                print(f"  {item['file']}")
                print(f"    Issues: {', '.join(item['issues'])}")

        print("=" * 60)


def create_clean_metadata(
    metadata_path: str, valid_files: List[str], output_path: str
) -> int:
    """
    Create cleaned metadata with only valid files

    Args:
        metadata_path: Original metadata path
        valid_files: List of valid file paths
        output_path: Output metadata path

    Returns:
        Number of entries in cleaned metadata
    """
    valid_files_set = set(valid_files)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        rows = list(reader)

    clean_rows = [row for row in rows if len(row) > 0 and row[0] in valid_files_set]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(clean_rows)

    return len(clean_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Validate audio dataset for XTTS v2 training"
    )

    # Input
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata CSV file'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        help='Base path for audio files (if not absolute in metadata)'
    )

    # Validation parameters
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=22050,
        help='Expected sample rate (default: 22050)'
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=1.0,
        help='Minimum duration in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=15.0,
        help='Maximum duration in seconds (default: 15.0)'
    )

    # Output options
    parser.add_argument(
        '--output_report',
        type=str,
        help='Save detailed report to JSON file'
    )
    parser.add_argument(
        '--fix_issues',
        action='store_true',
        help='Create cleaned metadata with only valid files'
    )
    parser.add_argument(
        '--output_fixed',
        type=str,
        help='Output path for cleaned metadata (requires --fix_issues)'
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Initialize validator
    validator = AudioValidator(
        target_sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    # Validate dataset
    results = validator.validate_dataset(args.metadata, args.base_path)

    # Print report
    validator.print_report(results)

    # Save detailed report
    if args.output_report:
        with open(args.output_report, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {args.output_report}")

    # Create cleaned metadata
    if args.fix_issues:
        if not args.output_fixed:
            print("\nError: --output_fixed is required when using --fix_issues")
            sys.exit(1)

        clean_count = create_clean_metadata(
            args.metadata, results['valid_files'], args.output_fixed
        )
        print(f"\nCleaned metadata saved to: {args.output_fixed}")
        print(f"Entries in cleaned metadata: {clean_count}")

    # Exit with error if validation found issues
    if results['invalid_samples'] > 0:
        print(f"\n⚠️  Warning: Found {results['invalid_samples']} invalid files")
        if not args.fix_issues:
            print("   Use --fix_issues --output_fixed to create cleaned metadata")
        sys.exit(1)
    else:
        print(f"\n✓ All files validated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
