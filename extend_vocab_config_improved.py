#!/usr/bin/env python3
"""
Improved Vocabulary Extension and Configuration for XTTS v2
Platform-independent implementation with text normalization and quality metrics

Usage:
    python extend_vocab_config_improved.py --output_path checkpoints/ --metadata_path datasets/metadata_train.csv --language be --extended_vocab_size 4000
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def normalize_belarusian_text(text: str) -> str:
    """
    Normalize Belarusian text before tokenization

    Args:
        text: Input text string

    Returns:
        Normalized text string
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Normalize apostrophes to consistent type
    text = text.replace("'", "'").replace("`", "'").replace("'", "'")

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"').replace('„', '"')

    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')

    return text


def normalize_text_for_language(text: str, language: str) -> str:
    """
    Normalize text based on language

    Args:
        text: Input text
        language: Language code (e.g., 'be', 'vi', 'en')

    Returns:
        Normalized text
    """
    if language == 'be':
        return normalize_belarusian_text(text)
    elif language == 'vi':
        # Vietnamese specific normalization could go here
        text = ' '.join(text.split())
        return text
    else:
        # Default normalization
        text = ' '.join(text.split())
        return text


def combine_tokenizers_platform_independent(
    old_tokenizer_path: str, new_tokenizer_path: str, save_dir: str
) -> None:
    """
    Combine two tokenizers in a platform-independent way

    Args:
        old_tokenizer_path: Path to old tokenizer directory
        new_tokenizer_path: Path to new tokenizer directory
        save_dir: Output directory for merged tokenizer
    """
    # Load vocabulary files
    with open(os.path.join(old_tokenizer_path, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab_old = json.load(f)

    with open(os.path.join(new_tokenizer_path, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab_new = json.load(f)

    # Create merged vocabulary
    merged_vocab = {}
    idx = 0

    # Add words from old tokenizer
    for word in vocab_old.keys():
        if word not in merged_vocab:
            merged_vocab[word] = idx
            idx += 1

    # Add words from new tokenizer
    for word in vocab_new.keys():
        if word not in merged_vocab:
            merged_vocab[word] = idx
            idx += 1

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Save merged vocabulary
    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_vocab, f, ensure_ascii=False, indent=2)

    # Merge merges files using Python (platform-independent)
    old_merges_path = os.path.join(old_tokenizer_path, 'merges.txt')
    new_merges_path = os.path.join(new_tokenizer_path, 'merges.txt')
    output_merges_path = os.path.join(save_dir, 'merges.txt')

    with open(output_merges_path, 'w', encoding='utf-8') as outfile:
        # Copy old merges
        with open(old_merges_path, 'r', encoding='utf-8') as infile:
            outfile.write(infile.read())

        # Append new merges (skip header line)
        with open(new_merges_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            if lines:
                # Skip first line (header)
                outfile.writelines(lines[1:])


def analyze_tokenization_quality(
    tokenizer: Tokenizer, texts: List[str], language: str
) -> Tuple[float, float, Dict[str, int]]:
    """
    Analyze tokenizer quality for the target language

    Args:
        tokenizer: Tokenizer to analyze
        texts: List of text samples
        language: Language code

    Returns:
        Tuple of (compression_ratio, unk_ratio, statistics)
    """
    total_chars = 0
    total_tokens = 0
    unk_count = 0
    total_samples = len(texts)

    for text in texts:
        encoding = tokenizer.encode(text)
        tokens = encoding.ids

        total_chars += len(text)
        total_tokens += len(tokens)
        unk_count += tokens.count(1)  # UNK token ID is typically 1

    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0

    statistics = {
        'total_samples': total_samples,
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'unk_count': unk_count,
        'avg_tokens_per_sample': total_tokens / total_samples if total_samples > 0 else 0,
    }

    return compression_ratio, unk_ratio, statistics


def recommend_vocab_size(dataset_hours: float, num_unique_words: int) -> int:
    """
    Recommend vocabulary size based on dataset characteristics

    Args:
        dataset_hours: Estimated dataset size in hours
        num_unique_words: Number of unique words in dataset

    Returns:
        Recommended vocabulary size
    """
    if dataset_hours < 10:
        base_size = 2000
    elif dataset_hours < 50:
        base_size = 3500
    else:
        base_size = 6000

    # Adjust based on vocabulary richness
    adjusted_size = min(base_size + (num_unique_words // 100), 8000)
    return adjusted_size


def validate_belarusian_text(text: str) -> Tuple[bool, str]:
    """
    Validate Belarusian text quality

    Args:
        text: Text to validate

    Returns:
        Tuple of (is_valid, reason)
    """
    # Check length
    if len(text) < 3 or len(text) > 500:
        return False, "invalid_length"

    # Check for valid Belarusian characters
    # Belarusian Cyrillic: а-я, ё, і, ў, apostrophe, common punctuation
    valid_chars_pattern = r'^[а-яёіўА-ЯЁІЎ\s\'\-\,\.\!\?\:\;\"0-9]+$'
    if not re.match(valid_chars_pattern, text):
        return False, "invalid_characters"

    # Check for reasonable word count
    words = text.split()
    if len(words) < 1:
        return False, "no_words"

    # Check for excessive repetition
    if len(words) > 1 and len(set(words)) < len(words) * 0.3:
        return False, "excessive_repetition"

    return True, "valid"


def clean_directory(directory: str) -> None:
    """
    Remove directory and its contents in a platform-independent way

    Args:
        directory: Directory path to remove
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)


def extend_tokenizer(args):
    """
    Extend tokenizer with new language vocabulary

    Args:
        args: Command line arguments
    """
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")

    print("=" * 60)
    print("XTTS v2 Vocabulary Extension")
    print("=" * 60)
    print(f"Language: {args.language}")
    print(f"Extended vocab size: {args.extended_vocab_size}")
    print(f"Metadata path: {args.metadata_path}")
    print("=" * 60)

    # Load and process training data
    print("\n[1/7] Loading training data...")
    traindf = pd.read_csv(args.metadata_path, sep="|")
    texts = traindf.text.to_list()
    print(f"  Loaded {len(texts)} text samples")

    # Normalize texts
    print(f"\n[2/7] Normalizing {args.language} text...")
    original_texts = texts.copy()
    texts = [normalize_text_for_language(text, args.language) for text in texts]

    # Validate texts for Belarusian
    if args.language == 'be':
        print("\n[3/7] Validating Belarusian texts...")
        invalid_count = 0
        for text in texts[:100]:  # Sample validation
            is_valid, reason = validate_belarusian_text(text)
            if not is_valid:
                invalid_count += 1
        if invalid_count > 0:
            print(f"  Warning: Found {invalid_count}/100 texts with issues")
    else:
        print("\n[3/7] Skipping text validation (only for Belarusian)")

    # Calculate unique words
    all_words = ' '.join(texts).split()
    unique_words = len(set(all_words))
    print(f"  Unique words in dataset: {unique_words}")

    # Recommend vocab size if auto
    if args.auto_vocab_size:
        recommended_size = recommend_vocab_size(args.dataset_hours, unique_words)
        print(f"  Recommended vocab size: {recommended_size}")
        if args.extended_vocab_size != recommended_size:
            print(f"  Using specified size: {args.extended_vocab_size}")

    # Save old tokenizer
    print("\n[4/7] Saving existing tokenizer...")
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)
    print(f"  Saved to: {old_tokenizer_path}")

    # Train new tokenizer
    print("\n[5/7] Training new tokenizer...")
    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=[f"[{args.language}]"],
        vocab_size=args.extended_vocab_size
    )
    new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
    new_tokenizer.add_special_tokens([f"[{args.language}]"])

    new_tokenizer_path = os.path.join(root, "new_tokenizer/")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)
    print(f"  New tokenizer trained with {args.extended_vocab_size} tokens")

    # Merge tokenizers
    print("\n[6/7] Merging tokenizers...")
    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    combine_tokenizers_platform_independent(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )

    # Load merged tokenizer and save to final location
    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(
        os.path.join(merged_tokenizer_path, 'vocab.json'),
        os.path.join(merged_tokenizer_path, 'merges.txt')
    )
    tokenizer.add_special_tokens([f"[{args.language}]"])
    tokenizer.save(os.path.join(root, "vocab.json"))
    print(f"  Merged tokenizer saved to: {os.path.join(root, 'vocab.json')}")

    # Analyze tokenization quality
    print("\n[7/7] Analyzing tokenization quality...")
    compression_ratio, unk_ratio, stats = analyze_tokenization_quality(
        tokenizer, texts[:1000], args.language  # Sample for speed
    )

    print(f"\nTokenization Quality Metrics:")
    print(f"  Compression ratio: {compression_ratio:.2f} chars/token")
    print(f"  UNK token ratio:   {unk_ratio*100:.2f}%")
    print(f"  Avg tokens/sample: {stats['avg_tokens_per_sample']:.1f}")
    print(f"\n  Quality Assessment:")
    if compression_ratio >= 3.0:
        print(f"    ✓ Good compression ratio (>= 3.0)")
    else:
        print(f"    ⚠ Low compression ratio (< 3.0) - consider larger vocab")
    if unk_ratio < 0.01:
        print(f"    ✓ Low UNK ratio (< 1%)")
    else:
        print(f"    ⚠ High UNK ratio (>= 1%) - may need larger vocab")

    # Cleanup temporary directories
    print("\nCleaning up temporary files...")
    clean_directory(old_tokenizer_path)
    clean_directory(new_tokenizer_path)
    clean_directory(merged_tokenizer_path)

    print("\n" + "=" * 60)
    print("Vocabulary extension completed successfully!")
    print("=" * 60)


def adjust_config(args):
    """
    Adjust XTTS configuration to include new language

    Args:
        args: Command line arguments
    """
    config_path = os.path.join(
        args.output_path, "XTTS_v2.0_original_model_files/config.json"
    )

    print(f"\nUpdating configuration file: {config_path}")

    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)

    # Add language if not already present
    if args.language not in config.get("languages", []):
        if "languages" not in config:
            config["languages"] = []
        config["languages"].append(args.language)
        print(f"  Added language '{args.language}' to config")
    else:
        print(f"  Language '{args.language}' already in config")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("  Configuration updated successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Extend XTTS v2 vocabulary for new language (Platform-Independent)"
    )

    # Required arguments
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to checkpoint directory containing XTTS_v2.0_original_model_files/"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to training metadata CSV file"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code (e.g., 'be' for Belarusian, 'vi' for Vietnamese)"
    )

    # Vocabulary size arguments
    parser.add_argument(
        "--extended_vocab_size",
        type=int,
        default=4000,
        help="Extended vocabulary size (default: 4000)"
    )
    parser.add_argument(
        "--auto_vocab_size",
        action="store_true",
        help="Automatically recommend vocabulary size based on dataset"
    )
    parser.add_argument(
        "--dataset_hours",
        type=float,
        default=20.0,
        help="Estimated dataset size in hours (for auto vocab size calculation)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.metadata_path):
        print(f"Error: Metadata file not found: {args.metadata_path}")
        return 1

    xtts_dir = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")
    if not os.path.exists(xtts_dir):
        print(f"Error: XTTS directory not found: {xtts_dir}")
        print("Please run download_checkpoint.py first")
        return 1

    # Run vocabulary extension
    try:
        extend_tokenizer(args)
        adjust_config(args)
        return 0
    except Exception as e:
        print(f"\nError during vocabulary extension: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
