import numpy as np
import torch
from tqdm import tqdm

from args import init_parser, init_sub_args
from dataset_padding import get_ttm_dataset_and_loader
from utils.data_utils import trans_list
from utils.model_padding import PoseTTMVAD
from utils.padding_trainer import TTMVADTrainer


torch.backends.mha.set_fastpath_enabled(False)


# ============================================================
# Load checkpoint first
# ============================================================

def load_checkpoint_info(
    checkpoint_path,
):
    # Safe here because this is your own trusted checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_config" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_config."
        )

    return checkpoint


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = init_parser()
    args = init_sub_args(
        parser.parse_args()
    )

    if args.model_ckpt_dir in (None, ""):
        raise ValueError(
            "--model_ckpt_dir is required."
        )

    device = torch.device(
        args.device
    )

    print("=" * 70)
    print("TEST-SET STATIC PROTOTYPE USAGE")
    print("=" * 70)

    print(
        "Checkpoint:",
        args.model_ckpt_dir,
    )

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Read checkpoint configuration
    #
    # Do this BEFORE creating the dataset, because seg_len,
    # stride, and TTM sequence length must match training.
    # --------------------------------------------------------

    checkpoint_info = load_checkpoint_info(
        args.model_ckpt_dir
    )

    config = checkpoint_info[
        "model_config"
    ]

    # Dataset-related configuration
    args.seg_len = int(
        config.get(
            "seg_len",
            args.seg_len,
        )
    )

    args.seg_stride = int(
        config.get(
            "seg_stride",
            args.seg_stride,
        )
    )

    args.ttm_sequence_length = int(
        config.get(
            "ttm_sequence_length",
            args.ttm_sequence_length,
        )
    )

    # --------------------------------------------------------
    # Dataset / test loader
    # --------------------------------------------------------

    _, dataset, loader = (
        get_ttm_dataset_and_loader(
            args,
            trans_list=trans_list,
            only_test=True,
        )
    )

    test_loader = loader[
        "test"
    ]

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    num_keypoints = int(
        config["num_keypoints"]
    )

    seg_len = int(
        config["seg_len"]
    )

    input_dim = int(
        config.get(
            "input_dim",
            2,
        )
    )

    embed_dim = int(
        config["embed_dim"]
    )

    process_size = int(
        config["process_size"]
    )

    memory_size = int(
        config["memory_size"]
    )

    num_layers = int(
        config["num_layers"]
    )

    mlp_dim = int(
        config["mlp_dim"]
    )

    num_heads = int(
        config["num_heads"]
    )

    window_num_layers = int(
        config["window_num_layers"]
    )

    dropout = float(
        config.get(
            "dropout",
            0.1,
        )
    )

    num_static_slots = int(
        config["num_static_slots"]
    )

    lambda_memory = float(
        config.get(
            "lambda_memory",
            0.05,
        )
    )

    lambda_static = float(
        config.get(
            "lambda_static",
            0.1,
        )
    )

    memory_score_weight = float(
        config.get(
            "memory_score_weight",
            1.0,
        )
    )

    static_score_weight = float(
        config.get(
            "static_score_weight",
            1.0,
        )
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = PoseTTMVAD(
        num_keypoints=num_keypoints,
        seg_len=seg_len,
        input_dim=input_dim,
        embed_dim=embed_dim,
        process_size=process_size,
        memory_size=memory_size,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout=dropout,
        memory_mode="TL",
        use_positional_embedding=True,
        num_static_slots=num_static_slots,
        window_num_layers=window_num_layers,
    ).to(
        device
    )

    # --------------------------------------------------------
    # Trainer only used for checkpoint loading and batch prep
    # --------------------------------------------------------

    trainer = TTMVADTrainer(
        model=model,
        train_loader=None,
        test_loader=test_loader,
        optimizer=None,
        scheduler=None,
        lambda_memory=lambda_memory,
        lambda_static=lambda_static,
        memory_score_weight=
            memory_score_weight,
        static_score_weight=
            static_score_weight,
        grad_clip=None,
    )

    # --------------------------------------------------------
    # Load BEST checkpoint
    #
    # This restores:
    #
    # model weights
    # static prototype bank
    # calibration thresholds
    # --------------------------------------------------------

    checkpoint = trainer.load_checkpoint(
        path=args.model_ckpt_dir,
        device=device,
        load_optimizer=False,
        load_scheduler=False,
    )

    model.eval()

    print()
    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Best AUC:",
        checkpoint.get(
            "best_auc",
            "unknown",
        ),
    )

    print(
        "Reconstruction threshold:",
        trainer.recon_gate_threshold,
    )

    print(
        "Static threshold:",
        trainer.static_gate_threshold,
    )

    # --------------------------------------------------------
    # Training-set unused mask stored in checkpoint
    # --------------------------------------------------------

    train_unused_mask = checkpoint.get(
        "unused_prototype_mask",
        None,
    )

    if train_unused_mask is not None:

        if torch.is_tensor(
            train_unused_mask
        ):
            train_unused_mask = (
                train_unused_mask
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

        else:
            train_unused_mask = (
                np.asarray(
                    train_unused_mask
                )
                .astype(bool)
            )

        print()
        print(
            "Training calibration unused prototypes:",
            np.where(
                train_unused_mask
            )[0],
        )

    # ========================================================
    # TEST-SET INFERENCE
    # ========================================================

    prototype_assignments = []

    total_real_windows = 0

    with torch.inference_mode():

        for data_batch in tqdm(
            test_loader,
            desc="Inference / prototype usage",
        ):

            (
                data,
                valid_mask,
                _,
            ) = trainer._prepare_batch(
                data_batch,
                device,
            )

            # ------------------------------------------------
            # Match actual inference behavior.
            #
            # Use calibrated thresholds so dynamic-memory
            # evolution is the same as normal evaluation.
            # ------------------------------------------------

            if (
                trainer.recon_gate_threshold
                is not None
                and
                trainer.static_gate_threshold
                is not None
            ):

                output = model(
                    data,
                    valid_mask=valid_mask,
                    memory_update_mode="hard",
                    recon_threshold=
                        trainer.recon_gate_threshold,
                    static_threshold=
                        trainer.static_gate_threshold,
                )

            else:

                print(
                    "WARNING: thresholds unavailable. "
                    "Using memory_update_mode='always'."
                )

                output = model(
                    data,
                    valid_mask=valid_mask,
                    memory_update_mode="always",
                )

            # [B,S]
            static_assignment = output[
                "static_assignment"
            ]

            # -----------------------------------------------
            # Keep REAL windows only.
            #
            # Padding assignments are ignored.
            # -----------------------------------------------

            valid_assignment = (
                static_assignment[
                    valid_mask
                ]
            )

            if valid_assignment.numel() > 0:

                valid_assignment_np = (
                    valid_assignment
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )

                prototype_assignments.append(
                    valid_assignment_np
                )

                total_real_windows += (
                    len(
                        valid_assignment_np
                    )
                )

    # ========================================================
    # Concatenate assignments
    # ========================================================

    if len(
        prototype_assignments
    ) == 0:

        raise RuntimeError(
            "No valid prototype assignments "
            "were collected from the test set."
        )

    prototype_assignments = (
        np.concatenate(
            prototype_assignments,
            axis=0,
        )
        .astype(np.int64)
    )

    # Safety
    if np.any(
        prototype_assignments < 0
    ):
        raise RuntimeError(
            "Found -1 prototype assignment "
            "for a valid test window."
        )

    # ========================================================
    # Count test usage
    # ========================================================

    prototype_counts = np.bincount(
        prototype_assignments,
        minlength=num_static_slots,
    )

    total_assignments = int(
        prototype_counts.sum()
    )

    test_unused_mask = (
        prototype_counts == 0
    )

    num_unused = int(
        test_unused_mask.sum()
    )

    num_used = (
        num_static_slots
        -
        num_unused
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("=" * 70)
    print("TEST-SET STATIC PROTOTYPE USAGE")
    print("=" * 70)

    print(
        "Real test windows:",
        total_real_windows,
    )

    print(
        "Total assignments:",
        total_assignments,
    )

    print()

    for prototype_idx in range(
        num_static_slots
    ):

        count = int(
            prototype_counts[
                prototype_idx
            ]
        )

        percentage = (
            100.0
            *
            count
            /
            max(
                total_assignments,
                1,
            )
        )

        status = (
            "UNUSED"
            if count == 0
            else "USED"
        )

        print(
            f"Prototype {prototype_idx:2d}: "
            f"{count:7d} windows | "
            f"{percentage:7.3f}% | "
            f"{status}"
        )

    print()
    print(
        "Used prototypes:",
        num_used,
        "/",
        num_static_slots,
    )

    print(
        "Unused prototypes:",
        num_unused,
        "/",
        num_static_slots,
    )

    print(
        "Unused prototype indices:",
        np.where(
            test_unused_mask
        )[0],
    )

    # ========================================================
    # Compare TRAIN calibration usage with TEST usage
    # ========================================================

    if train_unused_mask is not None:

        if len(
            train_unused_mask
        ) != num_static_slots:

            raise RuntimeError(
                "Training unused mask size does "
                "not match static memory size."
            )

        # Prototype was unused on normal training data,
        # but test samples selected it.
        train_unused_test_used = (
            train_unused_mask
            &
            (~test_unused_mask)
        )

        print()
        print("=" * 70)
        print("TRAIN vs TEST PROTOTYPE USAGE")
        print("=" * 70)

        print(
            "Unused during training calibration:",
            np.where(
                train_unused_mask
            )[0],
        )

        print(
            "Unused during test:",
            np.where(
                test_unused_mask
            )[0],
        )

        print(
            "Unused on TRAIN but selected on TEST:",
            np.where(
                train_unused_test_used
            )[0],
        )


if __name__ == "__main__":
    main()