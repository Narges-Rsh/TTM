import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


# ============================================================
# Configuration
# ============================================================

# CSV files contain:
# ground_truth, raw_score, final_score
input_dir = Path(
    "/home/nrashvan/Token/results/ShanghaiTech_s24_st3_t8_m256_d16/best_scores/"
)

# Anomaly threshold
# Higher TTM score = more anomalous
threshold =0.242970 

# Gaussian smoothing
sigma = 7

# Binary temporal smoothing
window_size = 24
stride =3
# Short predicted-event filtering
apply_filter_short_events = True
min_duration = 3

# Event-level tIoU thresholds
tiou_thresholds = [
    0.2,
    0.3,
    0.4,
    0.5,
]


# ============================================================
# Gaussian smoothing
# ============================================================

def apply_gaussian_smoothing(
    scores,
    sigma_val=7,
):
    """
    Same iterative Gaussian smoothing used in the previous code.
    """

    s = scores.copy()

    for sig in range(
        1,
        sigma_val,
    ):
        s = gaussian_filter1d(
            s,
            sigma=sig,
        )

    return s


# ============================================================
# Binary temporal smoothing
# ============================================================

def apply_flexible_smoothing(
    series,
    window=24,
    stride=6,
):
    """
    Same binary majority-vote smoothing used previously.

    Convention:
        1 = anomaly
        0 = normal
    """

    arr = np.asarray(
        series,
        dtype=int,
    ).copy()

    n = len(arr)

    output = np.zeros(
        n,
        dtype=int,
    )

    offset = stride

    # --------------------------------------------------------
    # Full windows
    # --------------------------------------------------------

    for i in range(
        0,
        n - window + 1,
        stride,
    ):
        block = arr[
            i:i + window
        ]

        winner = (
            1
            if np.sum(block == 1)
            >
            np.sum(block == 0)
            else 0
        )

        start_idx = (
            i
            +
            offset
        )

        end_idx = (
            start_idx
            +
            stride
        )

        output[
            start_idx:end_idx
        ] = winner

    # --------------------------------------------------------
    # Tail
    # --------------------------------------------------------

    if n >= window:

        last_processed_end = (
            ((n - window) // stride) * stride
            +
            offset
            +
            stride
        )

    else:

        last_processed_end = 0

    if last_processed_end < n:

        remaining_block = arr[
            last_processed_end:
        ]

        if len(remaining_block) > 0:

            winner = (
                1
                if np.sum(remaining_block == 1)
                >
                np.sum(remaining_block == 0)
                else 0
            )

            output[
                last_processed_end:
            ] = winner

    return output


# ============================================================
# Convert binary sequence to events
# ============================================================

def get_intervals(
    array,
):
    """
    Extract contiguous intervals where value == 1.

    Example:

        [0,0,1,1,1,0,1,1]

    becomes:

        [(2,4), (6,7)]
    """

    array = np.asarray(
        array,
        dtype=int,
    )

    padded = np.concatenate(
        [
            [0],
            array,
            [0],
        ]
    )

    diff = np.diff(
        padded
    )

    starts = np.where(
        diff == 1
    )[0]

    ends = (
        np.where(
            diff == -1
        )[0]
        -
        1
    )

    return list(
        zip(
            starts,
            ends,
        )
    )


# ============================================================
# Temporal IoU
# ============================================================

def calculate_tiou(
    int_a,
    int_b,
):
    s1, e1 = int_a
    s2, e2 = int_b

    intersection = max(
        0,
        min(e1, e2)
        -
        max(s1, s2)
        +
        1,
    )

    union = (
        (e1 - s1 + 1)
        +
        (e2 - s2 + 1)
        -
        intersection
    )

    return (
        intersection / union
        if union > 0
        else 0.0
    )


# ============================================================
# Process all files
# ============================================================

all_gt = []
all_pred = []


print("=" * 70)
print("EVENT-LEVEL VAD EVALUATION")
print("=" * 70)

print(
    f"Threshold          : {threshold}"
)

print(
    f"Gaussian sigma     : {sigma}"
)

print(
    f"Binary window      : {window_size}"
)

print(
    f"Binary stride      : {stride}"
)

print(
    f"Minimum duration   : > {min_duration} frames"
)

print("=" * 70)


csv_files = sorted(
    input_dir.glob("*.csv")
)


if len(csv_files) == 0:

    raise RuntimeError(
        f"No CSV files found in {input_dir}"
    )


for csv_file in csv_files:

    # ========================================================
    # 1. Load current TTM CSV
    # ========================================================

    df = pd.read_csv(
        csv_file
    )
    

    

    required_columns = {
        "ground_truth",
        "raw_score",
    }

    missing_columns = (
        required_columns
        -
        set(df.columns)
    )

    if missing_columns:

        raise KeyError(
            f"{csv_file.name} is missing columns: "
            f"{missing_columns}"
        )


    # ========================================================
    # 2. Read GT and RAW score
    #
    # IMPORTANT:
    # raw_score is used because Gaussian smoothing is performed
    # below, exactly as in the old processing pipeline.
    # ========================================================

    gt = pd.to_numeric(
        df["ground_truth"],
        errors="raise",
    ).astype(int).to_numpy()

    raw_score = pd.to_numeric(
        df["raw_score"],
        errors="raise",
    ).astype(float).to_numpy()


    if not np.all(
        np.isfinite(raw_score)
    ):

        raise RuntimeError(
            f"Non-finite raw scores in "
            f"{csv_file.name}"
        )

    
    # ========================================================
    # 3. Gaussian smoothing
    # ========================================================

    smoothed_score = (
        apply_gaussian_smoothing(
            raw_score,
            sigma_val=sigma,
        )
    )


    # ========================================================
    # 4. Threshold
    #
    # Same behavior as your old script:
    #
    # score > threshold -> anomaly = 1
    # ========================================================

    binarized = np.where(
        smoothed_score
        >
        threshold,
        1,
        0,
    )


    # ========================================================
    # 5. Binary temporal smoothing
    # ========================================================

    smoothed_binarized = (
        apply_flexible_smoothing(
            binarized,
            window=window_size,
            stride=stride,
        )
    )


    # ========================================================
    # 6. Convert binary frame labels to events
    # ========================================================

    gt_intervals = get_intervals(
        gt
    )

    pred_intervals = get_intervals(
        smoothed_binarized
    )


    # ========================================================
    # 7. Remove short predicted events
    #
    # IMPORTANT:
    # Preserve old behavior:
    #
    #     duration > min_duration
    #
    # Therefore an event of exactly 24 frames is removed.
    # ========================================================

    if apply_filter_short_events:

        pred_intervals = [

            p

            for p in pred_intervals

            if (
                p[1]
                -
                p[0]
                +
                1
            )
            >
            min_duration
        ]


    all_gt.append(
        gt_intervals
    )

    all_pred.append(
        pred_intervals
    )


# ============================================================
# Dataset-level statistics
# ============================================================

total_gt_count = sum(
    len(events)
    for events in all_gt
)

total_pred_count = sum(
    len(events)
    for events in all_pred
)


print()
print("=" * 70)
print("DATASET TOTALS")
print("=" * 70)

print(
    f"Number of videos:             "
    f"{len(csv_files)}"
)

print(
    f"Total Ground Truth Events:    "
    f"{total_gt_count}"
)

print(
    f"Total Predicted Events:       "
    f"{total_pred_count}"
)

print(
    f"Event Count Error (Pred-GT):  "
    f"{total_pred_count - total_gt_count}"
)

print("=" * 70)


# ============================================================
# Event-level evaluation
# ============================================================

precisions_per_threshold = []
recalls_per_threshold = []
f1_scores_per_threshold = []


print()
print(
    "Calculating event-level metrics..."
)

print(
    f"{'tIoU':<6} | "
    f"{'TP':<5} | "
    f"{'FP':<5} | "
    f"{'FN':<5} | "
    f"{'Precision':<10} | "
    f"{'Recall':<10} | "
    f"{'F1':<10}"
)

print("-" * 75)


for tiou_thresh in tiou_thresholds:

    total_tp = 0


    # ========================================================
    # Match events independently inside each video
    # ========================================================

    for (
        gt_intervals,
        pred_intervals,
    ) in zip(
        all_gt,
        all_pred,
    ):

        matched_pred_indices = set()

        file_tp = 0


        for gt_interval in gt_intervals:

            best_iou = 0.0
            best_pred_idx = -1


            for (
                pred_idx,
                pred_interval,
            ) in enumerate(
                pred_intervals
            ):

                if (
                    pred_idx
                    in
                    matched_pred_indices
                ):
                    continue


                iou = calculate_tiou(
                    gt_interval,
                    pred_interval,
                )


                if iou > best_iou:

                    best_iou = iou
                    best_pred_idx = pred_idx


            if (
                best_iou
                >=
                tiou_thresh
            ):

                file_tp += 1

                matched_pred_indices.add(
                    best_pred_idx
                )


        total_tp += file_tp


    # ========================================================
    # FP / FN
    # ========================================================

    total_fp = (
        total_pred_count
        -
        total_tp
    )

    total_fn = (
        total_gt_count
        -
        total_tp
    )


    # ========================================================
    # Precision
    # ========================================================

    precision = (
        total_tp
        /
        total_pred_count

        if total_pred_count > 0

        else 0.0
    )


    # ========================================================
    # Recall
    # ========================================================

    recall = (
        total_tp
        /
        total_gt_count

        if total_gt_count > 0

        else 0.0
    )


    # ========================================================
    # F1
    # ========================================================

    f1 = (

        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )

        if (
            precision
            +
            recall
        ) > 0

        else 0.0
    )


    precisions_per_threshold.append(
        precision
    )

    recalls_per_threshold.append(
        recall
    )

    f1_scores_per_threshold.append(
        f1
    )


    print(
        f"{tiou_thresh:<6.1f} | "
        f"{total_tp:<5} | "
        f"{total_fp:<5} | "
        f"{total_fn:<5} | "
        f"{precision:<10.4f} | "
        f"{recall:<10.4f} | "
        f"{f1:<10.4f}"
    )


# ============================================================
# Summary
# ============================================================

mean_precision = np.mean(
    precisions_per_threshold
)

mean_recall = np.mean(
    recalls_per_threshold
)

mean_f1 = np.mean(
    f1_scores_per_threshold
)


print()
print("=" * 70)
print("FINAL EVENT-LEVEL RESULTS")
print("=" * 70)

print(
    f"Mean Precision across tIoU thresholds: "
    f"{mean_precision:.4f}"
)

print(
    f"Mean Recall across tIoU thresholds:    "
    f"{mean_recall:.4f}"
)

print(
    f"Mean F1 across tIoU thresholds:        "
    f"{mean_f1:.4f}"
)

print("=" * 70)


# ============================================================
# Predicted-event duration statistics
# ============================================================

all_durations = [

    (
        event[1]
        -
        event[0]
        +
        1
    )

    for video_events
    in all_pred

    for event
    in video_events
]


if all_durations:

    print()
    print("-" * 40)
    print("PREDICTED EVENT STATISTICS")
    print("-" * 40)

    print(
        f"Shortest Event: "
        f"{min(all_durations)} frames"
    )

    print(
        f"Longest Event:  "
        f"{max(all_durations)} frames"
    )

    print(
        f"Average Event:  "
        f"{np.mean(all_durations):.2f} frames"
    )

    print("-" * 40)

else:

    print(
        "\nNo events were predicted with the "
        "current threshold/min_duration."
    )