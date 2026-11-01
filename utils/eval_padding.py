import os
import csv

import numpy as np

from scipy.ndimage import gaussian_filter1d

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
)

from tqdm import tqdm

from dataset_padding import shanghaitech_hr_skip


# ============================================================
# Target false-negative rate
# ============================================================

target_fnr = 0.1


# ============================================================
# SCORE COMPLETE DATASET
# ============================================================

def score_dataset(
    score,
    metadata,
    args=None,
    save_csv=False,
):
    """
    Convert segment/window-level anomaly scores into
    frame-level scores and calculate evaluation metrics.

    Parameters
    ----------
    score:
        One anomaly score per original pose segment/window.

    metadata:
        Original segment-level metadata.

    args:
        Experiment arguments.

    save_csv:
        False:
            do not save frame-level scores.

        True:
            save the FINAL SMOOTHED frame-level scores.

    Returns
    -------
    (
        AUC_ROC,
        AUC_PR,
        EER,
        EER_threshold,
        FPR_at_target_FNR,
        threshold_at_target_FNR,
    )
    """

    if args is None:

        raise ValueError(
            "args is required for score_dataset()."
        )


    # ========================================================
    # Convert segment scores -> frame scores
    # ========================================================

    (
        gt_arr,
        scores_arr,
        clip_names,

    ) = get_dataset_scores(

        score,
        metadata,
        args=args,
    )



    # ========================================================
    # Keep RAW frame-level scores before temporal smoothing
    # ========================================================

    raw_scores_arr = [
        clip_score.copy()
        for clip_score in scores_arr
    ]


    # ========================================================
    # Smooth FRAME-LEVEL scores
    #
    # IMPORTANT:
    #
    # Metrics below are calculated from these smoothed scores.
    #
    # Therefore CSV files are also saved AFTER this step.
    # ========================================================

    scores_arr = smooth_scores(
        scores_arr
    )


    # ========================================================
    # SAVE FINAL SCORES
    #
    # Only happens when:
    #
    # save_csv=True
    #
    # AND
    #
    # args.save_results=True
    #
    # In your trainer:
    #
    # reconstruction -> False
    # memory         -> False
    # static         -> False
    # combined       -> True
    #
    # Therefore only COMBINED final scores are saved.
    # ========================================================

    if (
        save_csv
        and
        getattr(
            args,
            "save_results",
            False,
        )
    ):

        os.makedirs(
            args.save_results_dir,
            exist_ok=True,
        )


        for (
            clip_name,
            clip_gt,
            clip_raw_score,
            clip_score,

        ) in zip(
            clip_names,
            gt_arr,
            raw_scores_arr,
            scores_arr,
        ):

            csv_path = os.path.join(
                args.save_results_dir,
                clip_name + ".csv",
            )


            with open(
                csv_path,
                "w",
                newline="",
            ) as f:

                writer = csv.writer(
                    f
                )


                # ------------------------------------------------
                # These are the EXACT smoothed scores that will
                # subsequently be used for metric calculation.
                # ------------------------------------------------

                writer.writerow(
                    [
                        "ground_truth",
                        "raw_score",
                        "final_score",
                    ]
                )


                writer.writerows(
                    zip(
                        clip_gt,
                        clip_raw_score,
                        clip_score,
                    )
                )


    # ========================================================
    # Flatten all clips for final dataset-level metrics
    # ========================================================

    gt_np = np.concatenate(
        gt_arr
    )


    scores_np = np.concatenate(
        scores_arr
    )


    # ========================================================
    # Calculate metrics
    # ========================================================

    (
        auc_roc,
        auc_precision_recall,
        EER,
        eer_threshold,
        fpr_at_target_fnr,
        threshold_at_target_fnr,

    ) = score_auc(

        scores_np,
        gt_np,
    )


    return (

        auc_roc,
        auc_precision_recall,
        EER,
        eer_threshold,
        fpr_at_target_fnr,
        threshold_at_target_fnr,
    )


def get_dataset_scores(
    scores,
    metadata,
    args=None,
):

    if args is None:

        raise ValueError(
            "args is required for get_dataset_scores()."
        )


    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_clip_names = []


    metadata_np = np.asarray(
        metadata
    )


    # ========================================================
    # Ground-truth files
    # ========================================================

    clip_list = os.listdir(
        args.mask_root
    )


    clip_list = sorted(

        fn

        for fn in clip_list

        if fn.endswith(
            ".npy"
        )
    )


    print(
        "Scoring {} clips".format(
            len(
                clip_list
            )
        )
    )


    # ========================================================
    # Score each clip
    #
    # IMPORTANT:
    #
    # Frames not covered by any pose window remain -inf
    # at this stage.
    #
    # They will be replaced later using the GLOBAL
    # minimum score across the entire dataset,
    # exactly like eval2.py.
    # ========================================================

    for clip in tqdm(
        clip_list
    ):

        (
            clip_gt,
            clip_score,

        ) = get_clip_score(

            scores,
            clip,
            metadata_np,
            metadata,
            args.mask_root,
            args,
        )


        if clip_score is None:

            continue


        dataset_gt_arr.append(
            clip_gt
        )


        dataset_scores_arr.append(
            clip_score
        )


        dataset_clip_names.append(

            clip.split(
                "."
            )[0]
        )


    # ========================================================
    # Safety
    # ========================================================

    if len(
        dataset_scores_arr
    ) == 0:

        raise RuntimeError(
            "No clips were scored."
        )


    # ========================================================
    # GLOBAL missing-frame replacement
    #
    # Same idea as eval2.py.
    #
    # Concatenate scores from ALL clips.
    # ========================================================

    scores_np = np.concatenate(
        dataset_scores_arr,
        axis=0,
    )


    # --------------------------------------------------------
    # Find all finite scores across the ENTIRE dataset
    # --------------------------------------------------------

    finite_scores = scores_np[
        np.isfinite(
            scores_np
        )
    ]


    if finite_scores.size == 0:

        raise RuntimeError(
            "No finite frame-level anomaly scores "
            "were found in the dataset."
        )


    # --------------------------------------------------------
    # Same behavior as eval2.py:
    #
    # +inf -> GLOBAL maximum finite score
    #
    # -inf -> GLOBAL minimum finite score
    # --------------------------------------------------------

    global_min_score = float(
        finite_scores.min()
    )

    global_max_score = float(
        finite_scores.max()
    )


    scores_np[
        scores_np == np.inf
    ] = global_max_score


    scores_np[
        scores_np == -np.inf
    ] = global_min_score


    # --------------------------------------------------------
    # NaN should not occur.
    # --------------------------------------------------------

    if np.any(
        np.isnan(
            scores_np
        )
    ):

        raise RuntimeError(
            "NaN frame-level anomaly scores "
            "were found."
        )


    print(
        "Global minimum anomaly score used "
        "for uncovered frames:",
        global_min_score
    )


    # ========================================================
    # Put globally corrected scores back into each clip
    #
    # This is equivalent to what eval2.py does with
    # its index loop.
    # ========================================================

    index = 0


    for clip_idx in range(
        len(
            dataset_scores_arr
        )
    ):

        clip_length = (
            dataset_scores_arr[
                clip_idx
            ].shape[0]
        )


        dataset_scores_arr[
            clip_idx
        ] = scores_np[
            index:
            index + clip_length
        ].copy()


        index += clip_length


    # ========================================================
    # Final safety
    # ========================================================

    if index != len(
        scores_np
    ):

        raise RuntimeError(
            "Error while restoring global scores "
            "back to individual clips."
        )


    # ========================================================
    # Do NOT save CSV here.
    #
    # The scores still need temporal smoothing.
    #
    # CSV saving happens later inside score_dataset()
    # after smoothing.
    # ========================================================

    return (

        dataset_gt_arr,
        dataset_scores_arr,
        dataset_clip_names,
    )








# ============================================================
# FINAL METRICS
# ============================================================

def score_auc(
    scores_np,
    gt,
):

    # ========================================================
    # Safety copy
    # ========================================================

    scores_np = np.asarray(
        scores_np,
        dtype=np.float64,
    ).copy()


    gt = np.asarray(
        gt
    )


    # ========================================================
    # Handle non-finite values if any remain
    #
    # Normally eval3 get_clip_score() already guarantees
    # finite frame scores.
    # ========================================================

    finite_mask = np.isfinite(
        scores_np
    )


    if not np.any(
        finite_mask
    ):

        raise RuntimeError(
            "No finite anomaly scores available "
            "for evaluation."
        )


    finite_scores = scores_np[
        finite_mask
    ]


    # Positive infinity -> largest finite score

    scores_np[
        scores_np == np.inf
    ] = finite_scores.max()


    # Negative infinity -> smallest finite score

    scores_np[
        scores_np == -np.inf
    ] = finite_scores.min()


    # NaN should not normally occur.
    #
    # Make this explicit instead of silently evaluating it.

    if np.any(
        np.isnan(
            scores_np
        )
    ):

        raise RuntimeError(
            "NaN anomaly scores found during evaluation."
        )


    # ========================================================
    # AUC-ROC
    # ========================================================

    auc_roc = roc_auc_score(
        gt,
        scores_np,
    )


    # ========================================================
    # AUC Precision-Recall
    # ========================================================

    (
        precision,
        recall,
        _,

    ) = precision_recall_curve(

        gt,
        scores_np,
    )


    auc_precision_recall = auc(
        recall,
        precision,
    )


    # ========================================================
    # ROC curve
    #
    # IMPORTANT:
    #
    # This still assumes label 1 is the POSITIVE class.
    #
    # We have NOT changed this yet.
    # We should separately verify your actual GT convention.
    # ========================================================

    (
        fpr,
        tpr,
        threshold,

    ) = roc_curve(

        gt,
        scores_np,

        pos_label=1,
    )


    fnr = (
        1
        -
        tpr
    )


    # ========================================================
    # Equal Error Rate
    # ========================================================

    eer_index = np.nanargmin(

        np.absolute(
            fnr
            -
            fpr
        )
    )


    eer_threshold = threshold[
        eer_index
    ]


    EER = fpr[
        eer_index
    ]


    # ========================================================
    # FPR when FNR is approximately 10%
    # ========================================================

    idx_closest_fnr = np.argmin(

        np.abs(
            fnr
            -
            target_fnr
        )
    )


    threshold_at_target_fnr = threshold[
        idx_closest_fnr
    ]


    fpr_at_target_fnr = fpr[
        idx_closest_fnr
    ]


    return (

        auc_roc,
        auc_precision_recall,
        EER,
        eer_threshold,
        fpr_at_target_fnr,
        threshold_at_target_fnr,
    )


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

def smooth_scores(
    scores_arr,
    sigma=7,
):
    """
    Current smoothing behavior is preserved from the old code.

    IMPORTANT:

    This applies smoothing repeatedly:

        sigma = 1
        sigma = 2
        ...
        sigma = 6

    when sigma=7.

    We are intentionally NOT changing this behavior yet.
    """

    for s in range(
        len(
            scores_arr
        )
    ):

        for sig in range(
            1,
            sigma,
        ):

            scores_arr[
                s
            ] = gaussian_filter1d(

                scores_arr[
                    s
                ],

                sigma=sig,
            )


    return scores_arr


# ============================================================
# SCORE ONE CLIP
# ============================================================

def get_clip_score(
    scores,
    clip,
    metadata_np,
    metadata,
    per_frame_scores_root,
    args,
):
    """
    Convert one score per pose segment/window into
    one anomaly score per VIDEO FRAME.

    Current TTM interpretation:

        ONE TTM step
            =
        ONE 24-frame pose window

    Therefore the segment anomaly score is assigned to
    the ENTIRE temporal interval represented by that segment.

    Overlapping windows:
        maximum score wins.

    Multiple people:
        maximum score wins.

    Frames with no pose-window coverage:
    temporarily remain -inf.

    After all clips are processed,
    get_dataset_scores() replaces these values
    with the minimum finite anomaly score across
    the ENTIRE evaluated dataset.

    This preserves the missing-frame behavior
    of eval2.py.
    """

    # ========================================================
    # Scene / clip IDs
    # ========================================================

    scene_id, clip_id = [

        int(i)

        for i in
        clip.split(
            "."
        )[0].split(
            "_"
        )[:2]
    ]


    # ========================================================
    # Skip clips according to ShanghaiTech-HR protocol
    # ========================================================

    if shanghaitech_hr_skip(

        (
            args.dataset
            ==
            "ShanghaiTech-HR"
        ),

        scene_id,
        clip_id,
    ):

        return (
            None,
            None,
        )


    # ========================================================
    # Find ALL pose segments belonging to this clip
    # ========================================================

    clip_metadata_inds = np.where(

        (
            metadata_np[:, 1]
            ==
            clip_id
        )

        &

        (
            metadata_np[:, 0]
            ==
            scene_id
        )

    )[0]


    # ========================================================
    # Load frame-level ground truth
    # ========================================================

    clip_res_fn = os.path.join(

        per_frame_scores_root,
        clip,
    )


    clip_gt = np.load(
        clip_res_fn
    )


    num_frames = (
        clip_gt.shape[0]
    )


    # ========================================================
    # Initialize frame-level anomaly score
    #
    # -inf:
    #     this frame has not yet received a segment score.
    # ========================================================

    clip_score = np.full(

        num_frames,

        -np.inf,

        dtype=np.float32,
    )


    # ========================================================
    # Map every segment/window score to frames
    # ========================================================

    for metadata_index in clip_metadata_inds:

        # ----------------------------------------------------
        # Segment start frame
        # ----------------------------------------------------

        start_frame = int(

            metadata[
                metadata_index
            ][3]
        )


        # ----------------------------------------------------
        # Segment interval
        #
        # For seg_len = 24:
        #
        # start=0
        #
        # covers frames:
        #
        # 0 ... 23
        # ----------------------------------------------------

        end_frame = (

            start_frame

            +

            int(
                args.seg_len
            )
        )


        # ====================================================
        # Keep interval inside video
        # ====================================================

        start_frame = max(
            start_frame,
            0,
        )


        end_frame = min(
            end_frame,
            num_frames,
        )


        if (
            end_frame
            <=
            start_frame
        ):

            continue


        # ====================================================
        # Segment anomaly score
        # ====================================================

        segment_score = float(

            scores[
                metadata_index
            ]
        )


        if not np.isfinite(
            segment_score
        ):

            raise RuntimeError(

                f"Non-finite segment score found "
                f"for metadata index "
                f"{metadata_index}."
            )


        # ====================================================
        # Assign this anomaly score to the ENTIRE window.
        #
        # If:
        #
        # - windows overlap
        # - different people overlap
        #
        # maximum anomaly score is retained.
        # ====================================================

        clip_score[
            start_frame:end_frame
        ] = np.maximum(

            clip_score[
                start_frame:end_frame
            ],

            segment_score,
        )


    





    # ========================================================
    # IMPORTANT:
    #
    # Frames that were not covered by ANY pose segment
    # intentionally remain -inf here.
    #
    # We DO NOT replace them with the minimum of this clip.
    #
    # get_dataset_scores() will later:
    #
    # 1. concatenate ALL clips
    # 2. find the minimum finite anomaly score across
    #    the ENTIRE dataset
    # 3. replace every remaining -inf with that
    #    GLOBAL minimum
    #
    # This preserves the behavior of eval2.py.
    # ========================================================


    # ========================================================
    # Shape safety
    # ========================================================

    if (
        clip_score.shape[0]
        !=
        clip_gt.shape[0]
    ):

        raise RuntimeError(

            f"Frame-score length does not match "
            f"ground-truth length for clip {clip}. "
            f"scores={clip_score.shape[0]}, "
            f"GT={clip_gt.shape[0]}"
        )


    return (

        clip_gt,
        clip_score,
    )