import numpy as np

from args import init_parser, init_sub_args
from dataset_padding import get_ttm_dataset_and_loader
from utils.data_utils import trans_list


def analyze_sequence_lengths(person_dataset, name="TRAIN"):
    """Print statistics for the original tracked-person sequences."""
    sequence_lengths = np.asarray(
        [
            len(track["segment_indices"])
            for track in person_dataset.tracks
        ],
        dtype=np.int64,
    )

    print()
    print("=" * 70)
    print(f"{name} DATASET - ORIGINAL WINDOWS PER PERSON")
    print("=" * 70)

    if len(sequence_lengths) == 0:
        print("No tracked-person sequences found.")
        return sequence_lengths

    sequence_length = person_dataset.sequence_length

    print("Number of person tracks:", len(sequence_lengths))
    print("Minimum S:", sequence_lengths.min())
    print("Maximum S:", sequence_lengths.max())
    print("Average S:", sequence_lengths.mean())
    print("Median S:", np.median(sequence_lengths))
    print("Standard deviation:", sequence_lengths.std())

    print("\nPercentiles:")
    for percentile in (25, 50, 75, 90, 95, 99):
        print(
            f"{percentile}%:",
            np.percentile(sequence_lengths, percentile),
        )

    print("\nTotal original windows:", sequence_lengths.sum())

    num_short = (sequence_lengths <= sequence_length).sum()
    pct_short = 100.0 * np.mean(sequence_lengths <= sequence_length)

    print(
        f"Tracks with S <= {sequence_length}:",
        num_short,
        "/",
        len(sequence_lengths),
        f"({pct_short:.2f}%)",
    )

    return sequence_lengths


def inspect_fixed_ttm_batch(data_loader, name="TRAIN"):
    """Inspect one batch produced by the fixed-length TTM loader."""
    if data_loader is None:
        return

    batch = next(iter(data_loader))

    print()
    print("=" * 70)
    print(f"{name} FIXED TTM BATCH")
    print("=" * 70)
    print("Pose shape:", batch["pose"].shape)
    print("Valid mask shape:", batch["valid_mask"].shape)
    print("Number of real steps:", batch["num_steps"])
    print("Track keys:", batch["track_key"])
    print("Chunk indices:", batch["chunk_index"])
    print("Chunk start steps:", batch["chunk_start_step"])
    print("Valid masks:\n", batch["valid_mask"])
    print("Segment indices:\n", batch["segment_indices"])
    print("Start frames:\n", batch["start_frames"])


def inspect_dataset_sample(person_dataset, index=0, name="TRAIN"):
    """Inspect one fixed-length person chunk."""
    sample = person_dataset[index]

    print()
    print("=" * 70)
    print(f"{name} FIXED-CHUNK SAMPLE {index}")
    print("=" * 70)
    print("Track:", sample["track_key"])
    print("Pose shape:", sample["pose"].shape)
    print("Valid real windows:", sample["num_steps"])
    print("Original person length:", sample["original_num_steps"])
    print("Chunk index:", sample["chunk_index"])
    print("Chunk start step:", sample["chunk_start_step"])
    print("Valid mask:", sample["valid_mask"])
    print("Start frames:", sample["start_frames"])
    print("Segment indices:", sample["segment_indices"])


def inspect_data_dimensions(person_dataset, index=0, name="TEST"):
    """Inspect the tensor dimensions that will be given to the TTM."""
    sample_pose = person_dataset[index]["pose"]

    if sample_pose.ndim != 4:
        raise ValueError(
            "Expected person-sequence pose [S,C,W,V]. "
            f"Received {tuple(sample_pose.shape)}"
        )

    sequence_length, channels, seg_len, num_keypoints = sample_pose.shape

    print()
    print("=" * 70)
    print(f"{name} TTM DATA DIMENSIONS")
    print("=" * 70)
    print("Sample shape:", sample_pose.shape)
    print("TTM steps:", sequence_length)
    print("Input channels:", channels)
    print("Frames per window:", seg_len)
    print("Keypoints per frame:", num_keypoints)
    print("Model uses channels 0:2 -> x,y")
    print("Padding is identified using valid_mask.")


def main():
    parser = init_parser()
    parser.add_argument(
        "--inspect_index",
        type=int,
        default=0,
        help="Dataset sample index to inspect.",
    )

    args = init_sub_args(parser.parse_args())

    _, dataset, loader = get_ttm_dataset_and_loader(
        args,
        trans_list=trans_list,
        only_test=False,
    )

    analyze_sequence_lengths(dataset["train"], name="TRAIN")
    analyze_sequence_lengths(dataset["test"], name="TEST")

    inspect_fixed_ttm_batch(loader["train"], name="TRAIN")
    inspect_fixed_ttm_batch(loader["test"], name="TEST")

    inspect_dataset_sample(
        dataset["train"],
        index=args.inspect_index,
        name="TRAIN",
    )

    inspect_dataset_sample(
        dataset["test"],
        index=args.inspect_index,
        name="TEST",
    )

    inspect_data_dimensions(
        dataset["test"],
        index=args.inspect_index,
        name="TEST",
    )


if __name__ == "__main__":
    main()
