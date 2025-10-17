#!/usr/bin/env python3
"""
Speaker-Aware Batch Sampling for XTTS Training
Ensures batch diversity by sampling from different speakers

Usage:
    from speaker_aware_sampler import SpeakerAwareSampler

    sampler = SpeakerAwareSampler(dataset, samples_per_speaker=2)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
"""

import random
import numpy as np
from typing import Iterator, List, Dict, Optional
from collections import defaultdict
import torch
from torch.utils.data import Sampler, Dataset
import warnings


class SpeakerAwareSampler(Sampler):
    """
    Speaker-aware batch sampler that ensures diversity in each batch

    Features:
    - Samples from different speakers in each batch
    - Prevents speaker dominance in batches
    - Supports stratified sampling by speaker
    - Optional speaker balancing across epochs
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        samples_per_speaker: int = 1,
        speaker_key: str = 'speaker_name',
        shuffle: bool = True,
        drop_last: bool = False,
        balance_speakers: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize speaker-aware sampler

        Args:
            dataset: Dataset with speaker information
            batch_size: Batch size
            samples_per_speaker: Max samples per speaker in a batch (default: 1)
            speaker_key: Key or attribute name for speaker ID in dataset
            shuffle: Shuffle samples within speakers
            drop_last: Drop last incomplete batch
            balance_speakers: Oversample minority speakers for balance
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_speaker = samples_per_speaker
        self.speaker_key = speaker_key
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.balance_speakers = balance_speakers
        self.seed = seed

        # Build speaker index
        self.speaker_to_indices = self._build_speaker_index()
        self.speakers = list(self.speaker_to_indices.keys())
        self.num_speakers = len(self.speakers)

        if self.num_speakers == 0:
            raise ValueError("No speakers found in dataset")

        # Calculate number of batches
        total_samples = len(dataset)
        self.num_batches = total_samples // batch_size
        if not drop_last and total_samples % batch_size != 0:
            self.num_batches += 1

        # Random state
        self.rng = np.random.RandomState(seed)

        # Print statistics
        self._print_statistics()

    def _build_speaker_index(self) -> Dict[str, List[int]]:
        """Build index mapping speakers to sample indices"""
        speaker_to_indices = defaultdict(list)

        for idx in range(len(self.dataset)):
            try:
                # Try to get speaker from dataset item
                item = self.dataset[idx]

                if isinstance(item, dict) and self.speaker_key in item:
                    speaker = item[self.speaker_key]
                elif hasattr(item, self.speaker_key):
                    speaker = getattr(item, self.speaker_key)
                else:
                    # Try accessing dataset attributes directly
                    if hasattr(self.dataset, 'samples'):
                        sample = self.dataset.samples[idx]
                        speaker = sample.get(self.speaker_key, 'unknown')
                    else:
                        speaker = 'unknown'

                speaker_to_indices[str(speaker)].append(idx)

            except Exception as e:
                warnings.warn(f"Error getting speaker for sample {idx}: {e}")
                speaker_to_indices['unknown'].append(idx)

        return dict(speaker_to_indices)

    def _print_statistics(self):
        """Print speaker distribution statistics"""
        print(f"\n{'='*60}")
        print(f"Speaker-Aware Sampler Statistics")
        print(f"{'='*60}")
        print(f"Total speakers: {self.num_speakers}")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Samples per speaker per batch: {self.samples_per_speaker}")
        print(f"Number of batches: {self.num_batches}")
        print(f"\nSpeaker distribution:")

        # Sort speakers by number of samples
        sorted_speakers = sorted(
            self.speaker_to_indices.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for speaker, indices in sorted_speakers[:10]:  # Show top 10
            count = len(indices)
            percentage = (count / len(self.dataset)) * 100
            print(f"  {speaker}: {count} samples ({percentage:.1f}%)")

        if len(sorted_speakers) > 10:
            print(f"  ... and {len(sorted_speakers) - 10} more speakers")

        # Check for imbalance
        counts = [len(indices) for indices in self.speaker_to_indices.values()]
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 10:
            print(f"\n⚠️  High speaker imbalance detected (ratio: {imbalance_ratio:.1f})")
            print(f"   Consider setting balance_speakers=True")

        print(f"{'='*60}\n")

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches with speaker diversity
        """
        # Create pools of indices for each speaker
        speaker_pools = {}
        for speaker, indices in self.speaker_to_indices.items():
            if self.shuffle:
                indices = indices.copy()
                self.rng.shuffle(indices)

            # Balance speakers if requested
            if self.balance_speakers:
                max_samples = max(len(v) for v in self.speaker_to_indices.values())
                # Oversample minority speakers
                while len(indices) < max_samples:
                    indices.extend(indices[:max_samples - len(indices)])
                if self.shuffle:
                    self.rng.shuffle(indices)

            speaker_pools[speaker] = indices

        # Generate batches
        for _ in range(self.num_batches):
            batch = []

            # Calculate how many speakers we need
            speakers_needed = min(
                self.num_speakers,
                (self.batch_size + self.samples_per_speaker - 1) // self.samples_per_speaker
            )

            # Randomly select speakers for this batch
            available_speakers = [
                s for s in self.speakers
                if len(speaker_pools[s]) > 0
            ]

            if len(available_speakers) == 0:
                # Refill pools
                for speaker, indices in self.speaker_to_indices.items():
                    if self.shuffle:
                        indices = indices.copy()
                        self.rng.shuffle(indices)
                    speaker_pools[speaker] = indices
                available_speakers = list(self.speakers)

            # Sample speakers
            selected_speakers = self.rng.choice(
                available_speakers,
                size=min(speakers_needed, len(available_speakers)),
                replace=False
            )

            # Sample from each speaker
            for speaker in selected_speakers:
                pool = speaker_pools[speaker]
                samples_to_take = min(self.samples_per_speaker, len(pool), self.batch_size - len(batch))

                for _ in range(samples_to_take):
                    if len(pool) > 0 and len(batch) < self.batch_size:
                        batch.append(pool.pop(0))

                if len(batch) >= self.batch_size:
                    break

            # Fill remaining batch slots if needed
            while len(batch) < self.batch_size:
                # Find any speaker with samples
                for speaker in self.speakers:
                    if len(speaker_pools[speaker]) > 0:
                        batch.append(speaker_pools[speaker].pop(0))
                        if len(batch) >= self.batch_size:
                            break
                else:
                    # No more samples available
                    break

            # Yield batch if it's complete or we don't drop last
            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                if self.shuffle:
                    self.rng.shuffle(batch)
                yield batch

    def __len__(self) -> int:
        """Return number of batches"""
        return self.num_batches


class StratifiedSpeakerSampler(Sampler):
    """
    Stratified speaker sampler ensuring equal representation of all speakers

    This sampler creates batches where each speaker is represented equally,
    useful when you want to ensure all speakers get equal training time.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        speaker_key: str = 'speaker_name',
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize stratified speaker sampler

        Args:
            dataset: Dataset with speaker information
            batch_size: Batch size
            speaker_key: Key for speaker ID in dataset
            shuffle: Shuffle samples
            drop_last: Drop last incomplete batch
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.speaker_key = speaker_key
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        # Build speaker index
        self.speaker_to_indices = self._build_speaker_index()
        self.speakers = list(self.speaker_to_indices.keys())
        self.num_speakers = len(self.speakers)

        # Calculate samples per speaker per batch
        self.samples_per_speaker = max(1, batch_size // self.num_speakers)

        # Random state
        self.rng = np.random.RandomState(seed)

    def _build_speaker_index(self) -> Dict[str, List[int]]:
        """Build speaker index"""
        speaker_to_indices = defaultdict(list)

        for idx in range(len(self.dataset)):
            try:
                item = self.dataset[idx]
                if isinstance(item, dict) and self.speaker_key in item:
                    speaker = item[self.speaker_key]
                elif hasattr(item, self.speaker_key):
                    speaker = getattr(item, self.speaker_key)
                else:
                    speaker = 'unknown'
                speaker_to_indices[str(speaker)].append(idx)
            except Exception:
                speaker_to_indices['unknown'].append(idx)

        return dict(speaker_to_indices)

    def __iter__(self) -> Iterator[List[int]]:
        """Generate stratified batches"""
        # Create iterators for each speaker
        speaker_iters = {}
        for speaker, indices in self.speaker_to_indices.items():
            if self.shuffle:
                indices = indices.copy()
                self.rng.shuffle(indices)
            speaker_iters[speaker] = iter(indices)

        while True:
            batch = []

            # Sample from each speaker
            for speaker in self.speakers:
                for _ in range(self.samples_per_speaker):
                    try:
                        idx = next(speaker_iters[speaker])
                        batch.append(idx)
                    except StopIteration:
                        # Restart this speaker's iterator
                        indices = self.speaker_to_indices[speaker].copy()
                        if self.shuffle:
                            self.rng.shuffle(indices)
                        speaker_iters[speaker] = iter(indices)

                        try:
                            idx = next(speaker_iters[speaker])
                            batch.append(idx)
                        except StopIteration:
                            pass

                    if len(batch) >= self.batch_size:
                        break

                if len(batch) >= self.batch_size:
                    break

            if len(batch) == 0:
                break

            if len(batch) == self.batch_size or not self.drop_last:
                if self.shuffle:
                    self.rng.shuffle(batch)
                yield batch[:self.batch_size]

    def __len__(self) -> int:
        """Return number of batches"""
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


def analyze_batch_speaker_diversity(
    dataloader,
    num_batches: int = 10,
    speaker_key: str = 'speaker_name',
) -> Dict:
    """
    Analyze speaker diversity in batches from a dataloader

    Args:
        dataloader: DataLoader to analyze
        num_batches: Number of batches to analyze
        speaker_key: Key for speaker information

    Returns:
        Dictionary with diversity statistics
    """
    batch_diversities = []
    speaker_counts = defaultdict(int)

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        # Get speakers in this batch
        if isinstance(batch, dict):
            speakers = batch.get(speaker_key, [])
        elif isinstance(batch, (list, tuple)):
            speakers = [item.get(speaker_key, 'unknown') for item in batch]
        else:
            continue

        # Count unique speakers
        unique_speakers = len(set(speakers))
        batch_diversities.append(unique_speakers)

        # Update global counts
        for speaker in speakers:
            speaker_counts[speaker] += 1

    return {
        "avg_unique_speakers_per_batch": np.mean(batch_diversities),
        "std_unique_speakers_per_batch": np.std(batch_diversities),
        "min_unique_speakers_per_batch": np.min(batch_diversities),
        "max_unique_speakers_per_batch": np.max(batch_diversities),
        "total_unique_speakers": len(speaker_counts),
        "speaker_counts": dict(speaker_counts),
    }


# Example usage
if __name__ == "__main__":
    print("Speaker-Aware Sampler Module")
    print("=" * 60)

    # Create mock dataset
    class MockDataset(Dataset):
        def __init__(self, num_samples=1000, num_speakers=10):
            self.samples = []
            for i in range(num_samples):
                speaker = f"speaker_{i % num_speakers}"
                self.samples.append({
                    'audio': torch.randn(16000),
                    'text': f"Sample {i}",
                    'speaker_name': speaker,
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    # Test dataset
    dataset = MockDataset(num_samples=100, num_speakers=5)

    print("\nTesting Speaker-Aware Sampler:")
    sampler = SpeakerAwareSampler(
        dataset,
        batch_size=8,
        samples_per_speaker=2,
        shuffle=True,
    )

    # Generate a few batches
    print("\nFirst 3 batches:")
    batch_iter = iter(sampler)
    for i in range(3):
        batch_indices = next(batch_iter)
        batch_speakers = [dataset[idx]['speaker_name'] for idx in batch_indices]
        unique_speakers = len(set(batch_speakers))
        print(f"\nBatch {i+1}:")
        print(f"  Indices: {batch_indices[:5]}... ({len(batch_indices)} total)")
        print(f"  Speakers: {batch_speakers}")
        print(f"  Unique speakers: {unique_speakers}")

    print("\n" + "=" * 60)
    print("Testing Stratified Speaker Sampler:")
    stratified_sampler = StratifiedSpeakerSampler(
        dataset,
        batch_size=10,
        shuffle=True,
    )

    # Generate a few batches
    print("\nFirst 2 batches:")
    batch_iter = iter(stratified_sampler)
    for i in range(2):
        batch_indices = next(batch_iter)
        batch_speakers = [dataset[idx]['speaker_name'] for idx in batch_indices]
        speaker_dist = {s: batch_speakers.count(s) for s in set(batch_speakers)}
        print(f"\nBatch {i+1}:")
        print(f"  Speaker distribution: {speaker_dist}")

    print("\n" + "=" * 60)
    print("Speaker-aware sampler module loaded successfully!")
    print("Use: from speaker_aware_sampler import SpeakerAwareSampler")
