import os
import random

import numpy as np
import torch
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter

from args import init_parser, init_sub_args
from dataset_padding import get_ttm_dataset_and_loader
from utils.data_utils import trans_list
from utils.model_padding import PoseTTMVAD
from utils.optim import init_optimizer, init_scheduler
from utils.padding_trainer import TTMVADTrainer

torch.backends.mha.set_fastpath_enabled(False)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_evaluation_summary(
    results,
    title="EVALUATION",
):
    names = {
        "reconstruction": "Recon",
        "memory_shift": "Memory",
        "static_memory": "Static",
        "combined": "Combined",
    }

    print(f"\n{title}")
    print("-" * 130)

    for key, label in names.items():

        (
            auc_roc,
            auc_pr,
            eer,
            eer_threshold,
            fpr_at_10_fnr,
            threshold_at_10_fnr,
        ) = results[key]

        print(
            f"{label:10s} | "
            f"AUC-ROC={auc_roc:.4f} | "
            f"AUC-PR={auc_pr:.4f} | "
            f"EER={eer:.4f} | "
            f"EER-Th={eer_threshold:.6f} | "
            f"10ER={fpr_at_10_fnr:.4f} | "
            f"10ER-Th={threshold_at_10_fnr:.6f}"
        )


def get_model_dimensions(dataset):
    """Read tensor dimensions without printing sample/debug information."""
    sample_pose = dataset["test"][0]["pose"]

    if sample_pose.ndim != 4:
        raise ValueError(
            "Expected person-sequence pose [S,C,W,V]. "
            f"Received {tuple(sample_pose.shape)}"
        )

    _, channels, seg_len, num_keypoints = sample_pose.shape

    if channels < 2:
        raise ValueError(
            "Pose data must contain at least x and y channels."
        )

    return seg_len, num_keypoints


def main():
    # ==========================================================
    # Arguments / reproducibility
    # ==========================================================
    parser = init_parser()
    args = init_sub_args(parser.parse_args())

    if args.relative:
        raise ValueError(
            "Current TTM experiment uses absolute XY pose only. "
            "Do not enable --relative."
        )

    set_random_seed(args.seed)

    device = torch.device(args.device)

    pretrained_model = args.model_ckpt_dir
    test_only = pretrained_model not in (None, "")


    # ==========================================================
    # Automatic experiment directory
    # ==========================================================

    if not test_only:

        run_name = (
            f"{args.dataset}"
            f"_s{args.seg_len}"
            f"_st{args.seg_stride}"
            f"_t{args.ttm_sequence_length}"
            f"_m{args.num_static_slots}"
            f"_d{args.ttm_embed_dim}"
        )

        args.model_save_dir = os.path.join(
            args.model_save_dir,
            run_name,
        )


    # Create experiment directory
    os.makedirs(
        args.model_save_dir,
        exist_ok=True,
    )


    # ==========================================================
    # Default directory for saved evaluation scores
    # ==========================================================

    if (
        args.save_results
        and args.save_results_dir is None
    ):
        args.save_results_dir = os.path.join(
            args.model_save_dir,
            "best_scores",
        )


    print("=" * 70)
    print("Pose-based VAD with Token Turing Machine")
    print("=" * 70)
    print("Mode:", "TEST ONLY" if test_only else "TRAIN")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Experiment directory:", args.model_save_dir)

    if args.save_results:
        print("Score directory:", args.save_results_dir)


    # ==========================================================
    # Dataset
    # ==========================================================
    _, dataset, loader = get_ttm_dataset_and_loader(
        args,
        trans_list=trans_list,
        only_test=test_only,
    )

    seg_len, num_keypoints = get_model_dimensions(dataset)

    # ==========================================================
    # Runtime configuration from args.py
    # ==========================================================
    ttm_embed_dim = args.ttm_embed_dim
    ttm_process_size = args.ttm_process_size
    ttm_memory_size = args.ttm_memory_size
    ttm_num_layers = args.ttm_num_layers
    ttm_mlp_dim = args.ttm_mlp_dim
    ttm_num_heads = args.ttm_num_heads
    window_num_layers = args.window_num_layers

    lambda_memory = args.lambda_memory
    lambda_static = args.lambda_static

    target_lambda_static = lambda_static
    static_lambda_start = args.static_lambda_start
    static_warmup_epochs = args.static_warmup_epochs

    memory_score_weight = args.memory_score_weight
    static_score_weight = args.static_score_weight
    grad_clip = args.grad_clip
    num_static_slots = args.num_static_slots
    normal_gate_quantile = args.normal_gate_quantile

    # ==========================================================
    # Test-only: restore saved model/scoring configuration
    # ==========================================================
    checkpoint_info = None
    saved_config = {}

    if test_only:
        checkpoint_info = torch.load(
            pretrained_model,
            map_location="cpu",
        )

        saved_config = checkpoint_info.get(
            "model_config",
            {},
        )

        ttm_embed_dim = saved_config.get(
            "embed_dim",
            ttm_embed_dim,
        )
        ttm_process_size = saved_config.get(
            "process_size",
            ttm_process_size,
        )
        ttm_memory_size = saved_config.get(
            "memory_size",
            ttm_memory_size,
        )
        ttm_num_layers = saved_config.get(
            "num_layers",
            ttm_num_layers,
        )
        ttm_mlp_dim = saved_config.get(
            "mlp_dim",
            ttm_mlp_dim,
        )
        ttm_num_heads = saved_config.get(
            "num_heads",
            ttm_num_heads,
        )
        window_num_layers = saved_config.get(
            "window_num_layers",
            window_num_layers,
        )

        num_static_slots = saved_config.get(
            "num_static_slots",
            num_static_slots,
        )

        lambda_memory = saved_config.get(
            "lambda_memory",
            lambda_memory,
        )
        lambda_static = saved_config.get(
            "lambda_static",
            lambda_static,
        )

        memory_score_weight = saved_config.get(
            "memory_score_weight",
            memory_score_weight,
        )
        static_score_weight = saved_config.get(
            "static_score_weight",
            static_score_weight,
        )

        normal_gate_quantile = saved_config.get(
            "normal_gate_quantile",
            normal_gate_quantile,
        )

        saved_seg_len = saved_config.get(
            "seg_len",
            seg_len,
        )
        saved_num_keypoints = saved_config.get(
            "num_keypoints",
            num_keypoints,
        )

        if saved_seg_len != seg_len:
            raise ValueError(
                "Checkpoint segment length does not match dataset. "
                f"Checkpoint={saved_seg_len}, dataset={seg_len}"
            )

        if saved_num_keypoints != num_keypoints:
            raise ValueError(
                "Checkpoint keypoint count does not match dataset. "
                f"Checkpoint={saved_num_keypoints}, "
                f"dataset={num_keypoints}"
            )

    # ==========================================================
    # Validation
    # ==========================================================
    if ttm_embed_dim % ttm_num_heads != 0:
        raise ValueError(
            "ttm_embed_dim must be divisible by ttm_num_heads."
        )

    if not test_only:
        if static_warmup_epochs < 1:
            raise ValueError(
                "static_warmup_epochs must be >= 1."
            )

        if static_warmup_epochs > args.epochs:
            raise ValueError(
                "static_warmup_epochs cannot exceed total epochs. "
                f"warmup={static_warmup_epochs}, epochs={args.epochs}"
            )

    # ==========================================================
    # Model configuration
    # ==========================================================
    model_config = {
        "num_keypoints": num_keypoints,
        "input_dim": 2,
        "seg_len": seg_len,
        "seg_stride": args.seg_stride,
        "ttm_sequence_length": args.ttm_sequence_length,
        "embed_dim": ttm_embed_dim,
        "process_size": ttm_process_size,
        "memory_size": ttm_memory_size,
        "num_layers": ttm_num_layers,
        "mlp_dim": ttm_mlp_dim,
        "num_heads": ttm_num_heads,
        "window_num_layers": window_num_layers,
        "dropout": args.dropout,
        "memory_mode": "TL",
        "lambda_memory": lambda_memory,
        "lambda_static": lambda_static,
        "static_lambda_start": static_lambda_start,
        "static_warmup_epochs": static_warmup_epochs,
        "memory_score_weight": memory_score_weight,
        "static_score_weight": static_score_weight,
        "num_static_slots": num_static_slots,
        "normal_gate_quantile": normal_gate_quantile,
        "grad_clip": grad_clip,
    }

    print("\nModel configuration:")
    for key, value in model_config.items():
        print(f"  {key:24s}: {value}")

    # ==========================================================
    # Model
    # ==========================================================
    model = PoseTTMVAD(
        num_keypoints=num_keypoints,
        seg_len=seg_len,
        input_dim=2,
        embed_dim=ttm_embed_dim,
        process_size=ttm_process_size,
        memory_size=ttm_memory_size,
        num_layers=ttm_num_layers,
        mlp_dim=ttm_mlp_dim,
        num_heads=ttm_num_heads,
        dropout=args.dropout,
        memory_mode="TL",
        use_positional_embedding=True,
        num_static_slots=num_static_slots,
        window_num_layers=window_num_layers,
    ).to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {num_parameters:,}")

    # ==========================================================
    # Test-only mode
    # ==========================================================
    if test_only:
        trainer = TTMVADTrainer(
            model=model,
            train_loader=None,
            test_loader=loader["test"],
            optimizer=None,
            scheduler=None,
            lambda_memory=lambda_memory,
            lambda_static=lambda_static,
            memory_score_weight=memory_score_weight,
            static_score_weight=static_score_weight,
            grad_clip=None,
        )

        checkpoint = trainer.load_checkpoint(
            path=pretrained_model,
            device=device,
            load_optimizer=False,
            load_scheduler=False,
        )

        print("\nCheckpoint epoch:", checkpoint.get("epoch", "unknown"))
        print("Saved best AUC:", checkpoint.get("best_auc", "unknown"))
        print(
            "Gating thresholds: "
            f"recon={trainer.recon_gate_threshold}, "
            f"static={trainer.static_gate_threshold}"
        )

        results = trainer.evaluate(
            device=device,
            args=args,
            save_scores=args.save_results,
        )

        print_evaluation_summary(
            results,
            title="TEST RESULTS",
        )
        return

    # ==========================================================
    # Save experiment configuration
    # ==========================================================
    arguments = vars(args).copy()
    arguments["ttm_config"] = model_config

    yaml_path = os.path.join(
        args.model_save_dir,
        "arguments.yaml",
    )

    with open(yaml_path, "w") as file:
        yaml.safe_dump(
            arguments,
            file,
            sort_keys=False,
        )

    # Logging
    # ==========================================================

    run_name = os.path.basename(
        os.path.normpath(
            args.model_save_dir
        )
    )

    wandb_run = wandb.init(
        project="Pose-TTM-VAD",
        name=run_name,
        config=arguments,
        dir=args.model_save_dir,
    )

    # Save W&B URL
    wandb_url = wandb_run.url

    print("W&B run:", wandb_url)

    with open(
        os.path.join(
            args.model_save_dir,
            "wandb_url.txt",
        ),
        "w",
    ) as f:
        f.write(wandb_url + "\n")


    writer = SummaryWriter(
        args.model_save_dir
    )

    # ==========================================================
    # Optimizer / scheduler
    # ==========================================================
    optimizer_f = init_optimizer(
        args.model_optimizer,
        lr=args.model_lr,
        weight_decay=args.model_weight_decay,
    )

    optimizer = optimizer_f(
        model.parameters()
    )

    scheduler_f = init_scheduler(
        args.sched,
        lr=args.model_lr,
        epochs=args.epochs,
        warmup=3,
    )

    scheduler = (
        scheduler_f(optimizer)
        if scheduler_f is not None
        else None
    )

    # ==========================================================
    # Trainer
    # ==========================================================
    trainer = TTMVADTrainer(
        model=model,
        train_loader=loader["train"],
        test_loader=loader["test"],
        optimizer=optimizer,
        scheduler=scheduler,
        lambda_memory=lambda_memory,
        lambda_static=lambda_static,
        memory_score_weight=memory_score_weight,
        static_score_weight=static_score_weight,
        grad_clip=grad_clip,
    )

    best_auc = -float("inf")
    best_epoch = -1

    best_checkpoint_path = os.path.join(
        args.model_save_dir,
        "ttm_best.pth.tar",
    )

    # ==========================================================
    # Training
    # ==========================================================
    for epoch in range(args.epochs):
        if static_warmup_epochs <= 1:
            current_lambda_static = target_lambda_static
        else:
            progress = min(
                epoch / (static_warmup_epochs - 1),
                1.0,
            )

            current_lambda_static = (
                static_lambda_start
                + progress
                * (
                    target_lambda_static
                    - static_lambda_start
                )
            )

        trainer.loss_function.lambda_static = (
            current_lambda_static
        )
        trainer.lambda_static = (
            current_lambda_static
        )

        train_stats = trainer.train_epoch(
            device=device
        )

        thresholds = trainer.calibrate_normal_thresholds(
            device=device,
            quantile=normal_gate_quantile,
        )

        results = trainer.evaluate(
            device=device,
            args=args,
            save_scores=False,
        )

        recon_auc = results["reconstruction"][0]
        memory_auc = results["memory_shift"][0]
        static_auc = results["static_memory"][0]
        combined_auc = results["combined"][0]

        current_lr = trainer.get_lr()

        # ------------------------------------------------------
        # Minimal console tracking
        # ------------------------------------------------------
        print()
        print("=" * 70)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"lr={current_lr:.3e} | "
            f"lambda_static={current_lambda_static:.4f}"
        )

        print(
            "Loss | "
            f"total={train_stats['loss']:.6f} | "
            f"recon={train_stats['reconstruction']:.6f} | "
            f"memory={train_stats['memory_shift']:.6f} | "
            f"static={train_stats['static_memory']:.6f}"
        )

        print(
            "Gate | "
            f"recon={thresholds['reconstruction']:.6f} | "
            f"static={thresholds['static']:.6f}"
        )

        print(
            "AUC-ROC | "
            f"recon={recon_auc:.4f} | "
            f"memory={memory_auc:.4f} | "
            f"static={static_auc:.4f} | "
            f"combined={combined_auc:.4f}"
        )

        # ------------------------------------------------------
        # TensorBoard
        # ------------------------------------------------------
        writer.add_scalar(
            "Train/Total_Loss",
            train_stats["loss"],
            epoch,
        )
        writer.add_scalar(
            "Train/Reconstruction",
            train_stats["reconstruction"],
            epoch,
        )
        writer.add_scalar(
            "Train/Memory_Shift",
            train_stats["memory_shift"],
            epoch,
        )
        writer.add_scalar(
            "Train/Static_Memory",
            train_stats["static_memory"],
            epoch,
        )
        writer.add_scalar(
            "Train/Lambda_Static",
            current_lambda_static,
            epoch,
        )

        writer.add_scalar(
            "AUC/Reconstruction",
            recon_auc,
            epoch,
        )
        writer.add_scalar(
            "AUC/Memory",
            memory_auc,
            epoch,
        )
        writer.add_scalar(
            "AUC/Static_Memory",
            static_auc,
            epoch,
        )
        writer.add_scalar(
            "AUC/Combined",
            combined_auc,
            epoch,
        )

        if current_lr is not None:
            writer.add_scalar(
                "Learning_Rate",
                current_lr,
                epoch,
            )

        # ------------------------------------------------------
        # Weights & Biases
        # ------------------------------------------------------
        wandb_metrics = {
            "epoch": epoch + 1,
            "train/total_loss": train_stats["loss"],
            "train/reconstruction": train_stats["reconstruction"],
            "train/memory_shift": train_stats["memory_shift"],
            "train/static_memory": train_stats["static_memory"],
            "train/lambda_static": current_lambda_static,
            "eval/reconstruction_auc": recon_auc,
            "eval/memory_auc": memory_auc,
            "eval/static_auc": static_auc,
            "eval/combined_auc": combined_auc,
            "threshold/reconstruction": thresholds["reconstruction"],
            "threshold/static": thresholds["static"],
        }

        if current_lr is not None:
            wandb_metrics["train/learning_rate"] = (
                current_lr
            )

        wandb.log(
            wandb_metrics,
            step=epoch,
        )

        # ------------------------------------------------------
        # Best checkpoint
        # ------------------------------------------------------
        if combined_auc > best_auc:
            best_auc = combined_auc
            best_epoch = epoch

            trainer.save_checkpoint(
                path=best_checkpoint_path,
                epoch=epoch,
                best_auc=best_auc,
                model_config=model_config,
            )

            wandb.run.summary["best_combined_auc"] = (
                best_auc
            )
            wandb.run.summary["best_epoch"] = (
                best_epoch + 1
            )

            print(
                f"New best model | "
                f"epoch={best_epoch + 1} | "
                f"combined AUC={best_auc:.4f}"
            )

        trainer.step_scheduler()

    # ==========================================================
    # Final best-model evaluation
    # ==========================================================
    writer.close()

    print()
    print("=" * 70)
    print(
        f"Training complete | "
        f"best epoch={best_epoch + 1} | "
        f"best combined AUC={best_auc:.4f}"
    )
    print("Checkpoint:", best_checkpoint_path)

    trainer.load_checkpoint(
        path=best_checkpoint_path,
        device=device,
        load_optimizer=False,
        load_scheduler=False,
    )

    if (
        args.save_results
        and args.save_results_dir is None
    ):
        args.save_results_dir = os.path.join(
            args.model_save_dir,
            "best_scores",
        )

    best_results = trainer.evaluate(
        device=device,
        args=args,
        save_scores=args.save_results,
    )

    print_evaluation_summary(
        best_results,
        title="BEST MODEL RESULTS",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
