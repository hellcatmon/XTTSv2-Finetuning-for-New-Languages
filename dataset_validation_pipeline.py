#!/usr/bin/env python3
"""
Automated Dataset Validation Pipeline for XTTS Training
Comprehensive validation and preprocessing pipeline for TTS datasets

Usage:
    python dataset_validation_pipeline.py --input_dir datasets/raw --output_dir datasets/validated

    # Or use programmatically:
    from dataset_validation_pipeline import DatasetValidationPipeline

    pipeline = DatasetValidationPipeline(input_dir="datasets/raw")
    results = pipeline.run()
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
import torchaudio
import pandas as pd
from datetime import datetime
import re


@dataclass
class ValidationResult:
    """Result of dataset validation"""
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    total_duration_hours: float = 0.0
    valid_duration_hours: float = 0.0
    speaker_count: int = 0
    speaker_distribution: Dict[str, int] = None
    audio_issues: Dict[str, int] = None
    text_issues: Dict[str, int] = None
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.speaker_distribution is None:
            self.speaker_distribution = {}
        if self.audio_issues is None:
            self.audio_issues = {}
        if self.text_issues is None:
            self.text_issues = {}
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class DatasetValidationPipeline:
    """
    Automated dataset validation and preprocessing pipeline

    Features:
    - Audio quality validation (6 checks)
    - Text quality validation (5 checks)
    - Speaker distribution analysis
    - Automatic fixing of common issues
    - Train/eval split generation
    - Comprehensive reporting
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        metadata_file: str = "metadata.csv",
        target_sample_rate: int = 22050,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        min_text_length: int = 3,
        max_text_length: int = 500,
        language: str = "be",
        fix_issues: bool = True,
        train_ratio: float = 0.95,
        verbose: bool = True,
    ):
        """
        Initialize validation pipeline

        Args:
            input_dir: Input directory with audio files and metadata
            output_dir: Output directory for validated dataset (optional)
            metadata_file: Name of metadata CSV file
            target_sample_rate: Target sample rate (default: 22050)
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            min_text_length: Minimum text length in characters
            max_text_length: Maximum text length in characters
            language: Target language code (e.g., 'be' for Belarusian)
            fix_issues: Attempt to fix common issues automatically
            train_ratio: Ratio of training data (default: 0.95)
            verbose: Print progress information
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "validated"
        self.metadata_file = metadata_file
        self.target_sample_rate = target_sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.language = language
        self.fix_issues = fix_issues
        self.train_ratio = train_ratio
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validation result
        self.result = ValidationResult()

    def run(self) -> ValidationResult:
        """
        Run complete validation pipeline

        Returns:
            ValidationResult with statistics and issues
        """
        if self.verbose:
            print("=" * 80)
            print("Dataset Validation Pipeline")
            print("=" * 80)
            print(f"Input directory: {self.input_dir}")
            print(f"Output directory: {self.output_dir}")
            print(f"Target language: {self.language}")
            print(f"Fix issues: {self.fix_issues}")
            print("=" * 80)

        # Step 1: Load metadata
        if self.verbose:
            print("\n[1/6] Loading metadata...")
        metadata = self._load_metadata()

        # Step 2: Validate audio files
        if self.verbose:
            print(f"\n[2/6] Validating {len(metadata)} audio files...")
        valid_samples = self._validate_audio_files(metadata)

        # Step 3: Validate text
        if self.verbose:
            print(f"\n[3/6] Validating text for {len(valid_samples)} samples...")
        valid_samples = self._validate_text(valid_samples)

        # Step 4: Analyze speaker distribution
        if self.verbose:
            print(f"\n[4/6] Analyzing speaker distribution...")
        self._analyze_speakers(valid_samples)

        # Step 5: Generate train/eval split
        if self.verbose:
            print(f"\n[5/6] Generating train/eval split...")
        self._generate_splits(valid_samples)

        # Step 6: Generate report
        if self.verbose:
            print(f"\n[6/6] Generating validation report...")
        self._generate_report()

        if self.verbose:
            print("\n" + "=" * 80)
            self._print_summary()
            print("=" * 80)

        return self.result

    def _load_metadata(self) -> List[Dict]:
        """Load metadata from CSV file"""
        metadata_path = self.input_dir / self.metadata_file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                metadata.append(row)

        self.result.total_samples = len(metadata)

        if self.verbose:
            print(f"   Loaded {len(metadata)} samples from {metadata_path}")

        return metadata

    def _validate_audio_files(self, metadata: List[Dict]) -> List[Dict]:
        """Validate audio files"""
        valid_samples = []
        audio_issues = {
            'file_not_found': 0,
            'clipping': 0,
            'excessive_silence': 0,
            'low_energy': 0,
            'dc_offset': 0,
            'invalid_duration': 0,
            'wrong_sample_rate': 0,
            'load_error': 0,
        }

        for i, sample in enumerate(metadata):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{len(metadata)}")

            audio_path = self.input_dir / sample.get('audio_file', '')

            # Check if file exists
            if not audio_path.exists():
                audio_issues['file_not_found'] += 1
                self.result.errors.append(f"File not found: {audio_path}")
                continue

            try:
                # Load audio
                wav, sr = torchaudio.load(str(audio_path))

                # Convert to mono
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)

                # Calculate duration
                duration = wav.shape[1] / sr
                self.result.total_duration_hours += duration / 3600

                # Validation checks
                issues = []

                # 1. Duration check
                if duration < self.min_duration or duration > self.max_duration:
                    issues.append('invalid_duration')
                    audio_issues['invalid_duration'] += 1

                # 2. Sample rate check
                if sr != self.target_sample_rate:
                    if self.fix_issues:
                        wav = torchaudio.functional.resample(wav, sr, self.target_sample_rate)
                        sr = self.target_sample_rate
                    else:
                        issues.append('wrong_sample_rate')
                        audio_issues['wrong_sample_rate'] += 1

                # 3. Clipping check
                max_amplitude = torch.max(torch.abs(wav)).item()
                if max_amplitude > 0.99:
                    issues.append('clipping')
                    audio_issues['clipping'] += 1
                    if self.fix_issues:
                        # Normalize
                        wav = wav / (max_amplitude + 1e-8) * 0.95

                # 4. Silence check
                silence_threshold = 0.01
                silence_ratio = (torch.abs(wav) < silence_threshold).float().mean().item()
                if silence_ratio > 0.5:
                    issues.append('excessive_silence')
                    audio_issues['excessive_silence'] += 1

                # 5. Energy check
                rms = torch.sqrt(torch.mean(wav ** 2)).item()
                if rms < 0.01:
                    issues.append('low_energy')
                    audio_issues['low_energy'] += 1

                # 6. DC offset check
                dc_offset = torch.mean(wav).item()
                if abs(dc_offset) > 0.1:
                    issues.append('dc_offset')
                    audio_issues['dc_offset'] += 1
                    if self.fix_issues:
                        wav = wav - dc_offset

                # Save fixed audio if issues were fixed
                if self.fix_issues and issues and duration >= self.min_duration and duration <= self.max_duration:
                    output_audio_path = self.output_dir / "audio" / audio_path.name
                    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(str(output_audio_path), wav, sr, bits_per_sample=16)
                    sample['audio_file'] = str(Path("audio") / audio_path.name)
                    sample['fixed'] = True

                # Add to valid samples if no critical issues
                if 'invalid_duration' not in issues:
                    valid_samples.append(sample)
                    self.result.valid_duration_hours += duration / 3600

            except Exception as e:
                audio_issues['load_error'] += 1
                self.result.errors.append(f"Error loading {audio_path}: {str(e)}")

        self.result.audio_issues = audio_issues
        self.result.valid_samples = len(valid_samples)

        if self.verbose:
            print(f"   Valid audio files: {len(valid_samples)}/{len(metadata)}")
            if sum(audio_issues.values()) > 0:
                print(f"   Audio issues found:")
                for issue, count in audio_issues.items():
                    if count > 0:
                        print(f"     - {issue}: {count}")

        return valid_samples

    def _validate_text(self, samples: List[Dict]) -> List[Dict]:
        """Validate text quality"""
        valid_samples = []
        text_issues = {
            'missing_text': 0,
            'invalid_length': 0,
            'invalid_characters': 0,
            'excessive_repetition': 0,
            'empty_after_normalization': 0,
        }

        for sample in samples:
            text = sample.get('text', '').strip()

            if not text:
                text_issues['missing_text'] += 1
                continue

            issues = []

            # 1. Length check
            if len(text) < self.min_text_length or len(text) > self.max_text_length:
                issues.append('invalid_length')
                text_issues['invalid_length'] += 1
                continue

            # 2. Character validation (language-specific)
            if self.language == 'be':
                # Belarusian: Cyrillic + common punctuation
                valid_pattern = r'^[а-яёіўА-ЯЁІЎ\s\'\-\,\.\!\?\:\;\"0-9]+$'
                if not re.match(valid_pattern, text):
                    if self.fix_issues:
                        # Try to clean text
                        text = self._normalize_belarusian_text(text)
                        if not re.match(valid_pattern, text):
                            issues.append('invalid_characters')
                            text_issues['invalid_characters'] += 1
                            continue
                    else:
                        issues.append('invalid_characters')
                        text_issues['invalid_characters'] += 1
                        continue

            # 3. Repetition check
            words = text.split()
            if len(words) > 0 and len(set(words)) < len(words) * 0.3:
                issues.append('excessive_repetition')
                text_issues['excessive_repetition'] += 1
                continue

            # 4. Normalize text if requested
            if self.fix_issues:
                original_text = text
                if self.language == 'be':
                    text = self._normalize_belarusian_text(text)

                if not text or len(text) < self.min_text_length:
                    issues.append('empty_after_normalization')
                    text_issues['empty_after_normalization'] += 1
                    continue

                if text != original_text:
                    sample['text'] = text
                    sample['text_normalized'] = True

            # Add to valid samples
            if not issues:
                valid_samples.append(sample)

        self.result.text_issues = text_issues
        self.result.valid_samples = len(valid_samples)

        if self.verbose:
            print(f"   Valid text samples: {len(valid_samples)}/{len(samples)}")
            if sum(text_issues.values()) > 0:
                print(f"   Text issues found:")
                for issue, count in text_issues.items():
                    if count > 0:
                        print(f"     - {issue}: {count}")

        return valid_samples

    def _normalize_belarusian_text(self, text: str) -> str:
        """Normalize Belarusian text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Normalize dashes
        text = text.replace('—', '-').replace('–', '-')

        # Remove any non-Belarusian characters
        text = re.sub(r'[^а-яёіўА-ЯЁІЎ\s\'\-\,\.\!\?\:\;\"0-9]', '', text)

        return text.strip()

    def _analyze_speakers(self, samples: List[Dict]):
        """Analyze speaker distribution"""
        speaker_distribution = {}

        for sample in samples:
            speaker = sample.get('speaker_name', 'unknown')
            speaker_distribution[speaker] = speaker_distribution.get(speaker, 0) + 1

        self.result.speaker_count = len(speaker_distribution)
        self.result.speaker_distribution = speaker_distribution

        if self.verbose:
            print(f"   Total speakers: {len(speaker_distribution)}")

            # Sort by sample count
            sorted_speakers = sorted(speaker_distribution.items(), key=lambda x: x[1], reverse=True)
            print(f"   Top speakers:")
            for speaker, count in sorted_speakers[:5]:
                percentage = (count / len(samples)) * 100
                print(f"     - {speaker}: {count} samples ({percentage:.1f}%)")

            # Check for imbalance
            if len(speaker_distribution) > 1:
                max_count = max(speaker_distribution.values())
                min_count = min(speaker_distribution.values())
                imbalance_ratio = max_count / min_count
                if imbalance_ratio > 10:
                    warning = f"High speaker imbalance detected (ratio: {imbalance_ratio:.1f})"
                    self.result.warnings.append(warning)
                    if self.verbose:
                        print(f"   ⚠️  {warning}")

    def _generate_splits(self, samples: List[Dict]):
        """Generate train/eval splits"""
        import random
        random.seed(42)

        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * self.train_ratio)
        train_samples = shuffled[:split_idx]
        eval_samples = shuffled[split_idx:]

        # Save splits
        self._save_metadata(train_samples, self.output_dir / "metadata_train.csv")
        self._save_metadata(eval_samples, self.output_dir / "metadata_eval.csv")

        if self.verbose:
            print(f"   Train samples: {len(train_samples)}")
            print(f"   Eval samples: {len(eval_samples)}")

    def _save_metadata(self, samples: List[Dict], output_path: Path):
        """Save metadata to CSV"""
        if not samples:
            return

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            # Use only essential fields
            fieldnames = ['audio_file', 'text', 'speaker_name']
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
            writer.writeheader()

            for sample in samples:
                writer.writerow({
                    'audio_file': sample.get('audio_file', ''),
                    'text': sample.get('text', ''),
                    'speaker_name': sample.get('speaker_name', 'unknown'),
                })

    def _generate_report(self):
        """Generate comprehensive validation report"""
        report_path = self.output_dir / "validation_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(self.input_dir),
            "output_directory": str(self.output_dir),
            "configuration": {
                "target_sample_rate": self.target_sample_rate,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration,
                "min_text_length": self.min_text_length,
                "max_text_length": self.max_text_length,
                "language": self.language,
                "fix_issues": self.fix_issues,
                "train_ratio": self.train_ratio,
            },
            "results": asdict(self.result),
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"   Report saved to: {report_path}")

    def _print_summary(self):
        """Print validation summary"""
        print("\nValidation Summary:")
        print(f"  Total samples: {self.result.total_samples}")
        print(f"  Valid samples: {self.result.valid_samples} ({self.result.valid_samples/self.result.total_samples*100:.1f}%)")
        print(f"  Invalid samples: {self.result.total_samples - self.result.valid_samples}")
        print(f"\n  Total duration: {self.result.total_duration_hours:.2f} hours")
        print(f"  Valid duration: {self.result.valid_duration_hours:.2f} hours")
        print(f"\n  Speakers: {self.result.speaker_count}")

        if self.result.warnings:
            print(f"\n  ⚠️  Warnings: {len(self.result.warnings)}")
            for warning in self.result.warnings[:5]:
                print(f"     - {warning}")

        if self.result.errors:
            print(f"\n  ❌ Errors: {len(self.result.errors)}")
            for error in self.result.errors[:5]:
                print(f"     - {error}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Validation Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with audio and metadata")
    parser.add_argument("--output_dir", type=str, help="Output directory for validated dataset")
    parser.add_argument("--metadata_file", type=str, default="metadata.csv", help="Metadata CSV filename")
    parser.add_argument("--target_sample_rate", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Minimum audio duration (seconds)")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Maximum audio duration (seconds)")
    parser.add_argument("--min_text_length", type=int, default=3, help="Minimum text length")
    parser.add_argument("--max_text_length", type=int, default=500, help="Maximum text length")
    parser.add_argument("--language", type=str, default="be", help="Target language code")
    parser.add_argument("--fix_issues", action="store_true", help="Attempt to fix common issues")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Training data ratio")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    # Run pipeline
    pipeline = DatasetValidationPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        target_sample_rate=args.target_sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        language=args.language,
        fix_issues=args.fix_issues,
        train_ratio=args.train_ratio,
        verbose=args.verbose,
    )

    result = pipeline.run()

    # Exit with error if validation failed significantly
    if result.valid_samples < result.total_samples * 0.5:
        print("\n❌ Validation failed: Less than 50% of samples are valid")
        exit(1)


if __name__ == "__main__":
    main()
