import os

import numpy as np
import torch

from tqdm import tqdm

from utils.eval_padding import score_dataset
from utils.loss_padding import TTMVADLoss
from sklearn.cluster import MiniBatchKMeans


class TTMVADTrainer:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        model,
        train_loader=None,
        test_loader=None,
        optimizer=None,
        scheduler=None,     
        lambda_memory=0.05,
        lambda_static=0.1,
        # lambda_diversity=0.0,
        memory_score_weight=1.0,
        static_score_weight=1.0,
        grad_clip=None,
        
    ):

        self.model = model

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_function = TTMVADLoss(
            lambda_memory=lambda_memory,
            lambda_static=lambda_static,
            # lambda_diversity=lambda_diversity,
        )

        self.lambda_memory = lambda_memory
        self.lambda_static = lambda_static
        # self.lambda_diversity = lambda_diversity

        self.memory_score_weight = (
            memory_score_weight
        )

        self.static_score_weight = (
            static_score_weight
        )

        self.grad_clip = grad_clip

        # Learned from NORMAL training data
        self.recon_gate_threshold = None
        self.static_gate_threshold = None
        
        self.prototype_counts = None
        self.unused_prototype_mask = None


    # ========================================================
    # PREPARE FIXED-LENGTH BATCH
    # ========================================================

    def _prepare_batch(
        self,
        data_batch,
        device,
    ):

        """
        Input:

            pose:
                [B,S,3,W,V]

            valid_mask:
                [B,S]

        Output:

            data:
                [B,S,2,W,V]

            valid_mask:
                [B,S]

            segment_indices:
                [B,S]
        """

        if "pose" not in data_batch:

            raise KeyError(
                "Expected data_batch['pose']."
            )

        if "valid_mask" not in data_batch:

            raise KeyError(
                "Expected data_batch['valid_mask']."
            )


        data = (
            data_batch["pose"]
            .to(
                device,
                non_blocking=True,
            )
        )


        if data.ndim != 5:

            raise ValueError(
                "Expected pose [B,S,C,W,V]. "
                f"Received {tuple(data.shape)}"
            )


        # ----------------------------------------------------
        # Keep X,Y only
        #
        # [B,S,3,W,V]
        #
        # ->
        #
        # [B,S,2,W,V]
        # ----------------------------------------------------

        data = data[
            :,
            :,
            0:2,
            :,
            :
        ].float()


        valid_mask = (
            data_batch["valid_mask"]
            .to(
                device,
                non_blocking=True,
            )
            .bool()
        )


        B, S = data.shape[:2]


        if tuple(
            valid_mask.shape
        ) != (B, S):

            raise ValueError(
                "valid_mask must match [B,S]. "
                f"Expected {(B, S)}, "
                f"received "
                f"{tuple(valid_mask.shape)}"
            )


        segment_indices = data_batch.get(
            "segment_indices",
            None,
        )


        return (
            data,
            valid_mask,
            segment_indices,
        )


    # ========================================================
    # GET ONE SCORE PER TTM STEP / 24-FRAME WINDOW
    # ========================================================

    def _get_step_scores(
        self,
        data,
        output,
        valid_mask,
    ):

        """
        data:
            [B,S,2,W,V]

        valid_mask:
            [B,S]

        Returns:

            reconstruction:
                [B,S]

            memory shift:
                [B,S]

            static distance:
                [B,S]

        Padding positions receive score 0.
        """

        # ====================================================
        # Reconstruction
        # ====================================================

        reconstruction = output[
            "reconstruction"
        ]


        if reconstruction.shape != data.shape:

            raise ValueError(
                "Reconstruction shape does not "
                "match target pose. "
                f"target={tuple(data.shape)}, "
                f"reconstruction="
                f"{tuple(reconstruction.shape)}"
            )


        recon = (

            (
                data
                -
                reconstruction
            )
            .pow(2)
            .mean(
                dim=(2, 3, 4)
            )
        )

        # [B,S]


        # ====================================================
        # Validate mask
        # ====================================================

        if valid_mask.shape != recon.shape:

            raise ValueError(
                "valid_mask must be [B,S]. "
                f"Expected {tuple(recon.shape)}, "
                f"received {tuple(valid_mask.shape)}"
            )


        mask_float = valid_mask.to(
            device=recon.device,
            dtype=recon.dtype,
        )


        # Padding gets zero reconstruction score

        recon = (
            recon
            *
            mask_float
        )


        # ====================================================
        # Dynamic-memory shift
        # ====================================================

        memory = output[
            "memory_shift_per_step"
        ]


        if memory.shape != recon.shape:

            raise ValueError(
                "memory_shift_per_step must be [B,S]. "
                f"Expected {tuple(recon.shape)}, "
                f"received {tuple(memory.shape)}"
            )


        memory = memory.clone()


        # ----------------------------------------------------
        # First transition:
        #
        # M0 -> M1
        #
        # Do not use it as anomaly score.
        # ----------------------------------------------------

        if memory.shape[1] > 0:

            memory[
                :,
                0
            ] = 0.0


        # Padding gets zero memory score

        memory = (
            memory
            *
            mask_float
        )


        # ====================================================
        # Static-memory distance
        # ====================================================

        static = output[
            "static_distance"
        ]


        if static.shape != recon.shape:

            raise ValueError(
                "static_distance must be [B,S]. "
                f"Expected {tuple(recon.shape)}, "
                f"received {tuple(static.shape)}"
            )


        # Padding gets zero static score

        static = (
            static
            *
            mask_float
        )


        return (
            recon,
            memory,
            static,
        )


    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train_epoch(
        self,
        device,
    ):

        if self.train_loader is None:

            raise ValueError(
                "train_loader is None."
            )


        if self.optimizer is None:

            raise ValueError(
                "optimizer is None."
            )


        self.model.train()


        # ====================================================
        # Running statistics
        # ====================================================

        running_loss = 0.0

        running_recon = 0.0
        running_memory = 0.0
        running_static = 0.0
        # running_diversity = 0.0

        # Number of independent chunks processed
        num_chunks = 0

        # Number of REAL memory transitions
        # num_memory_steps = 0


        for batch_idx, data_batch in enumerate( 
            tqdm(
            self.train_loader,
            desc="Training", )
        ):

            # =================================================
            # Prepare batch
            # =================================================

            (
                data,
                valid_mask,
                _,

            ) = self._prepare_batch(
                data_batch,
                device,
            )


            B, S, C, W, V = (
                data.shape
            )


            # =================================================
            # Clear gradients
            # =================================================

            self.optimizer.zero_grad(
                set_to_none=True
            )


            # =================================================
            # Forward
            #
            # valid_mask prevents padding from updating
            # the dynamic memory.
            # =================================================

            output = self.model(

                data,

                valid_mask=
                    valid_mask,

                memory_update_mode=
                    "always",
            )


            # =================================================
            # Loss
            #
            # valid_mask removes padding from loss.
            # =================================================

            losses = self.loss_function(

                data,

                output,

                valid_mask=
                    valid_mask,
            )


            loss = losses[
                "loss"
            ]


            # =================================================
            # Backpropagation
            # =================================================

            loss.backward()


            if self.grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )


            self.optimizer.step()


            # # =================================================
            # # Scores for statistics
            # # =================================================

            # (
            #     recon_step,
            #     memory_step,
            #     static_step,

            # ) = self._get_step_scores(

            #     data,

            #     output,

            #     valid_mask,
            # )


            # =================================================
            # Count REAL windows
            # =================================================

            # valid_steps = int(

            #     valid_mask
            #     .sum()
            #     .item()
            # )

            # num_steps += (
            #     valid_steps
            # )


            # # =================================================
            # # Count valid memory transitions
            # #
            # # valid:
            # #
            # # 1 1 1 1 0
            # #
            # # memory valid:
            # #
            # # 0 1 1 1 0
            # # =================================================

            # memory_valid_mask = (
            #     valid_mask.clone()
            # )


            # if S > 0:

            #     memory_valid_mask[
            #         :,
            #         0
            #     ] = False


            # num_memory_steps += int(

            #     memory_valid_mask
            #     .sum()
            #     .item()
            # )


           # ============================================================
            # Chunk-level training statistics
            #
            # The loss function already computes:
            #
            # reconstruction:
            #     [B]
            #
            # memory_shift:
            #     [B]
            #
            # static_memory:
            #     [B]
            #
            # Each value is ONE average value for ONE independent chunk.
            # ============================================================

            current_batch_size = data.shape[0]


            running_loss += (
                loss.item()
                *
                current_batch_size
            )


            running_recon += (
                losses[
                    "reconstruction"
                ]
                .detach()
                .sum()
                .item()
            )


            running_memory += (
                losses[
                    "memory_shift"
                ]
                .detach()
                .sum()
                .item()
            )


            running_static += (
                losses[
                    "static_memory"
                ]
                .detach()
                .sum()
                .item()
            )



            num_chunks += (
                current_batch_size
            )

            
            
           


        # ====================================================
        # Epoch statistics
        # ====================================================

        return {

            "loss":

                running_loss
                /
                max(
                    num_chunks,
                    1
                ),


            "reconstruction":

                running_recon
                /
                max(
                    num_chunks,
                    1
                ),


            "memory_shift":

                running_memory
                /
                max(
                    num_chunks,
                    1
                ),


            "static_memory":

                running_static
                /
                max(
                    num_chunks,
                    1
                ),


}
        
        
    # ========================================================
    # CALIBRATE NORMAL THRESHOLDS
    # ========================================================

    @torch.no_grad()
    def calibrate_normal_thresholds(
        self,
        device,
        quantile=0.99,
    ):

        if self.train_loader is None:

            raise ValueError(
                "train_loader is required "
                "for threshold calibration."
            )


        self.model.eval()


        recon_values = []
        static_values = []
        memory_values = []
        
        prototype_assignments = []


        for data_batch in tqdm(
            self.train_loader,
            desc="Calibrating normal thresholds",
        ):

            # =================================================
            # Prepare batch
            # =================================================

            (
                data,
                valid_mask,
                _,

            ) = self._prepare_batch(
                data_batch,
                device,
            )


            # =================================================
            # Forward
            # =================================================

            output = self.model(

                data,

                valid_mask=
                    valid_mask,

                memory_update_mode=
                    "always",
            )

            static_assignment = output[
                "static_assignment"
            ]

            # =================================================
            # Scores
            # =================================================

            (
                recon_step,
                memory_step,
                static_step,

            ) = self._get_step_scores(

                data,

                output,

                valid_mask,
            )


            # =================================================
            # ONLY collect REAL windows.
            #
            # Padding is completely ignored.
            # =================================================

            valid_recon = (
                recon_step[
                    valid_mask
                ]
            )


            memory_valid_mask = (
                valid_mask.clone()
            )

            if memory_valid_mask.shape[1] > 0:

                memory_valid_mask[
                    :,
                    0
                ] = False


            valid_memory = (
                memory_step[
                    memory_valid_mask
                ]
            )





            valid_static = (
                static_step[
                    valid_mask
                ]
            )
            
            # Prototype assignment for REAL windows only.
            # Padding assignments (-1) are excluded.

            valid_assignment = (
                static_assignment[
                    valid_mask
                ]
            )


            if valid_recon.numel() > 0:

                recon_values.append(

                    valid_recon
                    .cpu()
                    .numpy()
                )

            if valid_memory.numel() > 0:

                memory_values.append(
                    valid_memory
                    .cpu()
                    .numpy()
                )

            if valid_static.numel() > 0:

                static_values.append(

                    valid_static
                    .cpu()
                    .numpy()
                )
                
            if valid_assignment.numel() > 0:

                prototype_assignments.append(
                    valid_assignment
                    .detach()
                    .cpu()
                    .numpy()
                )


        # ====================================================
        # Safety
        # ====================================================

        if len(recon_values) == 0:

            raise RuntimeError(
                "No valid reconstruction values "
                "were collected."
            )


        if len(static_values) == 0:

            raise RuntimeError(
                "No valid static-memory values "
                "were collected."
            )


        # ====================================================
        # Concatenate real normal windows
        # ====================================================

        recon_values = np.concatenate(
            recon_values,
            axis=0,
        )


        if len(memory_values) > 0:

            memory_values = np.concatenate(
                memory_values,
                axis=0,
            )

        else:

            memory_values = np.asarray(
                [],
                dtype=np.float32,
            )


        static_values = np.concatenate(
            static_values,
            axis=0,
        )
        
        
        if len(prototype_assignments) == 0:

            raise RuntimeError(
                "No valid prototype assignments "
                "were collected."
            )

        
        
        
        prototype_assignments = np.concatenate(
            prototype_assignments,
            axis=0,
        ).astype(np.int64)



        if len(prototype_assignments) != len(static_values):

            raise RuntimeError(
                "Prototype assignments and static scores "
                "must have the same number of real windows."
            )



        num_static_slots = (
            self.model.static_memory.num_slots
        )

        prototype_counts = np.bincount(
            prototype_assignments,
            minlength=num_static_slots,
        )

        self.unused_prototype_mask = (
            prototype_counts == 0
        )

        unused_prototype_indices = np.where(
            self.unused_prototype_mask
        )[0]

        # ====================================================
        # Pairwise cosine similarity between static prototypes
        # ====================================================

        prototypes = (
            self.model
            .static_memory
            .prototypes
            .detach()
        )

        # [P, D]
        prototype_norms = (
            prototypes
            .norm(
                dim=-1,
                keepdim=True,
            )
            .clamp_min(
                1e-12
            )
        )

        normalized_prototypes = (
            prototypes
            /
            prototype_norms
        )

        # [P, P]
        prototype_similarity = (

            normalized_prototypes

            @

            normalized_prototypes.t()
        )


        prototype_similarity_np = (
            prototype_similarity
            .cpu()
            .numpy()
        )


        # Remove diagonal:
        #
        # similarity of each prototype with itself = 1
        pairwise_mask = ~np.eye(
            num_static_slots,
            dtype=bool,
        )


        off_diagonal_similarity = (
            prototype_similarity_np[
                pairwise_mask
            ]
        )


        print()
        print("=" * 70)
        print("STATIC PROTOTYPE PAIRWISE COSINE SIMILARITY")
        print("=" * 70)


        print(
            "Mean:",
            float(
                np.mean(
                    off_diagonal_similarity
                )
            )
        )

        print(
            "Std :",
            float(
                np.std(
                    off_diagonal_similarity
                )
            )
        )

        print(
            "Min :",
            float(
                np.min(
                    off_diagonal_similarity
                )
            )
        )

        print(
            "Max :",
            float(
                np.max(
                    off_diagonal_similarity
                )
            )
        )

        print(
            "P95 :",
            float(
                np.percentile(
                    off_diagonal_similarity,
                    95,
                )
            )
        )


        print()
        print("Prototype cosine-similarity matrix:")

        print(
            np.array2string(
                prototype_similarity_np,
                precision=3,
                suppress_small=True,
            )
        )


        print()
        print("Prototype vector norms:")

        print(
            prototypes
            .norm(
                dim=-1
            )
            .cpu()
            .numpy()
        )
        
        
        
        
        
        print()
        print("=" * 70)
        print("STATIC MEMORY PROTOTYPE USAGE")
        print("=" * 70)

        total_assignments = prototype_counts.sum()

        for prototype_idx in range(
            num_static_slots
        ):

            count = prototype_counts[
                prototype_idx
            ]

            percentage = (
                100.0
                *
                count
                /
                max(
                    total_assignments,
                    1
                )
            )

            print(
                f"Prototype {prototype_idx:2d}: "
                f"{count:5d} windows "
                f"({percentage:6.2f}%)"
            )


        used_prototypes = int(
            np.sum(
                prototype_counts > 0
            )
        )

        unused_prototypes = (
            num_static_slots
            -
            used_prototypes
        )

        print()

        print(
            "Used prototypes:",
            used_prototypes,
            "/",
            num_static_slots,
        )

        print(
            "Unused prototypes:",
            unused_prototypes,
            "/",
            num_static_slots,
        )



        print()
        print("=" * 70)
        print("STATIC MEMORY DISTANCE BY PROTOTYPE")
        print("=" * 70)

        for prototype_idx in range(
            num_static_slots
        ):

            prototype_mask = (
                prototype_assignments
                ==
                prototype_idx
            )

            count = prototype_mask.sum()

            if count == 0:

                print(
                    f"Prototype {prototype_idx:2d}: "
                    "UNUSED"
                )

                continue


            distances = static_values[
                prototype_mask
            ]


            print(
                f"Prototype {prototype_idx:2d}: "
                f"count={count:5d}, "
                f"mean_dist={np.mean(distances):.6f}, "
                f"std={np.std(distances):.6f}, "
                f"min={np.min(distances):.6f}, "
                f"max={np.max(distances):.6f}"
            )






        print()
        print("=" * 70)
        print("NORMAL TRAINING SCORE STATISTICS")
        print("=" * 70)


        def print_score_statistics(
            name,
            values,
        ):

            print()
            print(name)

            if len(values) == 0:

                print("  No valid values.")
                return

            print(
                "  Count :",
                len(values)
            )

            print(
                "  Mean  :",
                float(np.mean(values))
            )

            print(
                "  Std   :",
                float(np.std(values))
            )

            print(
                "  Min   :",
                float(np.min(values))
            )

            print(
                "  Max   :",
                float(np.max(values))
            )

            print(
                "  P95   :",
                float(np.percentile(
                    values,
                    95
                ))
            )

            print(
                "  P99   :",
                float(np.percentile(
                    values,
                    99
                ))
            )


        print_score_statistics(
            "RECONSTRUCTION",
            recon_values,
        )

        print_score_statistics(
            "MEMORY SHIFT",
            memory_values,
        )

        print_score_statistics(
            "STATIC MEMORY",
            static_values,
        )
                
        
        
        
        
        # ====================================================
        # Quantile thresholds
        # ====================================================

        self.recon_gate_threshold = float(

            np.quantile(
                recon_values,
                quantile,
            )
        )


        self.static_gate_threshold = float(

            np.quantile(
                static_values,
                quantile,
            )
        )


        print()

        print(
            "Normal calibration REAL windows:",
            len(recon_values)
        )


        print(
            "Reconstruction threshold:",
            self.recon_gate_threshold
        )


        print(
            "Static-memory threshold:",
            self.static_gate_threshold
        )


        return {

            "reconstruction":
                self.recon_gate_threshold,

            "static":
                self.static_gate_threshold,
        }


    # ========================================================
    # EVALUATION
    # ========================================================

    @torch.no_grad()
    def evaluate(
        self,
        device,
        args,
        save_scores=False,
    ):

        if self.test_loader is None:

            raise ValueError(
                "test_loader is None."
            )


        self.model.eval()


        # ====================================================
        # Original segment-level metadata
        # ====================================================

        if not hasattr(
            self.test_loader.dataset,
            "metadata",
        ):

            raise AttributeError(
                "PersonSequenceDataset must expose "
                "the original segment metadata."
            )


        metadata = (
            self.test_loader
            .dataset
            .metadata
        )


        num_segments = len(
            metadata
        )


        # ====================================================
        # Arrays in ORIGINAL segment order
        # ====================================================

        recon_scores = np.full(
            num_segments,
            np.nan,
            dtype=np.float32,
        )


        memory_scores = np.full(
            num_segments,
            np.nan,
            dtype=np.float32,
        )


        static_scores = np.full(
            num_segments,
            np.nan,
            dtype=np.float32,
        )


        combined_scores = np.full(
            num_segments,
            np.nan,
            dtype=np.float32,
        )


        # ====================================================
        # Evaluate fixed chunks
        # ====================================================

        for data_batch in tqdm(
            self.test_loader,
            desc="Evaluation",
        ):

            (
                data,
                valid_mask,
                segment_indices,

            ) = self._prepare_batch(
                data_batch,
                device,
            )


            if segment_indices is None:

                raise KeyError(
                    "Evaluation requires "
                    "segment_indices."
                )


            # =================================================
            # Are learned thresholds available?
            # =================================================

            thresholds_available = (

                self.recon_gate_threshold
                is not None

                and

                self.static_gate_threshold
                is not None
            )


            # =================================================
            # Forward
            # =================================================

            if thresholds_available:

                output = self.model(

                    data,

                    valid_mask=
                        valid_mask,

                    memory_update_mode=
                        "hard",

                    recon_threshold=
                        self.recon_gate_threshold,

                    static_threshold=
                        self.static_gate_threshold,
                )


            else:

                output = self.model(

                    data,

                    valid_mask=
                        valid_mask,

                    memory_update_mode=
                        "always",
                )


            # =================================================
            # Per-window scores
            # =================================================

            (
                recon,
                memory,
                static,

            ) = self._get_step_scores(

                data,

                output,

                valid_mask,
            )


            # =================================================
            # Convert segment indices to NumPy
            #
            # [B,S]
            # =================================================

            if torch.is_tensor(
                segment_indices
            ):

                segment_indices_np = (

                    segment_indices
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )

            else:

                segment_indices_np = (

                    np.asarray(
                        segment_indices
                    )
                    .astype(np.int64)
                )


            # =================================================
            # Convert valid mask
            # =================================================

            valid_mask_np = (

                valid_mask
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )


            # =================================================
            # Convert scores
            # =================================================

            recon_np = (

                recon
                .detach()
                .cpu()
                .numpy()
            )


            memory_np = (

                memory
                .detach()
                .cpu()
                .numpy()
            )


            static_np = (

                static
                .detach()
                .cpu()
                .numpy()
            )


            # =================================================
            # Validate dimensions
            # =================================================

            expected_shape = tuple(
                recon_np.shape
            )


            if (
                tuple(
                    segment_indices_np.shape
                )
                !=
                expected_shape
            ):

                raise ValueError(

                    "segment_indices and scores "
                    "must both be [B,S]. "

                    f"indices="
                    f"{segment_indices_np.shape}, "

                    f"scores="
                    f"{recon_np.shape}"
                )


            if (
                tuple(
                    valid_mask_np.shape
                )
                !=
                expected_shape
            ):

                raise ValueError(

                    "valid_mask and scores "
                    "must both be [B,S]. "

                    f"mask="
                    f"{valid_mask_np.shape}, "

                    f"scores="
                    f"{recon_np.shape}"
                )


            # =================================================
            # Valid positions:
            #
            # valid_mask = True
            #
            # AND
            #
            # segment index >= 0
            # =================================================

            if np.any(

                valid_mask_np

                &

                (
                    segment_indices_np
                    <
                    0
                )
            ):

                raise RuntimeError(
                    "A valid TTM position has "
                    "segment_index < 0."
                )


            valid_positions = (

                valid_mask_np

                &

                (
                    segment_indices_np
                    >=
                    0
                )
            )


            # =================================================
            # Extract REAL windows only
            # =================================================

            valid_segment_indices = (

                segment_indices_np[
                    valid_positions
                ]
            )


            valid_recon = (

                recon_np[
                    valid_positions
                ]
            )


            valid_memory = (

                memory_np[
                    valid_positions
                ]
            )


            valid_static = (

                static_np[
                    valid_positions
                ]
            )
            
            
            


            # =================================================
            # Combined anomaly score
            # =================================================

            valid_combined = (

                valid_recon

                +

                self.memory_score_weight
                *
                valid_memory

                +

                self.static_score_weight
                *
                valid_static
            )


            # =================================================
            # Safety
            # =================================================

            if np.any(

                valid_segment_indices
                >=
                num_segments
            ):

                raise RuntimeError(
                    "Found a segment index outside "
                    "the original dataset range."
                )


            # =================================================
            # Restore scores into ORIGINAL dataset order
            # =================================================
            already_scored = ~np.isnan(
                combined_scores[
                    valid_segment_indices
                ]
            )

            if np.any(
                already_scored
            ):

                duplicate_indices = (
                    valid_segment_indices[
                        already_scored
                    ]
                )

                raise RuntimeError(
                    "Duplicate original segment indices "
                    "found during evaluation. "
                    f"First duplicates: "
                    f"{duplicate_indices[:20]}"
                )


            recon_scores[
                valid_segment_indices
            ] = valid_recon


            memory_scores[
                valid_segment_indices
            ] = valid_memory


            static_scores[
                valid_segment_indices
            ] = valid_static


            combined_scores[
                valid_segment_indices
            ] = valid_combined


        # ====================================================
        # Verify every original segment received a score
        # ====================================================

        missing = np.where(

            np.isnan(
                combined_scores
            )

        )[0]


        if len(missing) > 0:

            raise RuntimeError(

                f"{len(missing)} original segments "
                "did not receive TTM scores. "

                f"First missing indices: "
                f"{missing[:20]}"
            )


        # ====================================================
        # Existing evaluation
        # ====================================================

        recon_metrics = score_dataset(
            recon_scores,
            metadata,
            args=args,
            save_csv=False,
        )


        memory_metrics = score_dataset(
            memory_scores,
            metadata,
            args=args,
            save_csv=False,
        )


        static_metrics = score_dataset(
            static_scores,
            metadata,
            args=args,
            save_csv=False,
        )


        combined_metrics = score_dataset(
            combined_scores,
            metadata,
            args=args,
            save_csv=save_scores,
        )


        # ====================================================
        # Return
        # ====================================================

        return {

            "reconstruction":
                recon_metrics,

            "memory_shift":
                memory_metrics,

            "static_memory":
                static_metrics,

            "combined":
                combined_metrics,


            "raw_reconstruction":
                recon_scores,

            "raw_memory":
                memory_scores,

            "raw_static":
                static_scores,

            "raw_combined":
                combined_scores,
        }


    # ========================================================
    # STEP SCHEDULER
    # ========================================================

    def step_scheduler(
        self,
    ):

        if self.scheduler is not None:

            self.scheduler.step()


    # ========================================================
    # CURRENT LEARNING RATE
    # ========================================================

    def get_lr(
        self,
    ):

        if self.optimizer is None:

            return None


        return self.optimizer.param_groups[
            0
        ]["lr"]


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        path,
        epoch,
        best_auc,
        model_config=None,
    ):

        state = {

            "epoch":
                epoch + 1,

            "state_dict":
                self.model.state_dict(),

            "best_auc":
                best_auc,

            "model_config":
                model_config,

            "recon_gate_threshold":
                self.recon_gate_threshold,

            "static_gate_threshold":
                self.static_gate_threshold,
                
                
            "unused_prototype_mask":
                torch.as_tensor(
                    self.unused_prototype_mask,
                    dtype=torch.bool,
                ),
        }


        if self.optimizer is not None:

            state[
                "optimizer"
            ] = self.optimizer.state_dict()


        if self.scheduler is not None:

            state[
                "scheduler"
            ] = self.scheduler.state_dict()


        save_dir = os.path.dirname(
            path
        )


        if save_dir:

            os.makedirs(
                save_dir,
                exist_ok=True,
            )


        torch.save(
            state,
            path,
        )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def load_checkpoint(
        self,
        path,
        device,
        load_optimizer=True,
        load_scheduler=True,
    ):

        checkpoint = torch.load(
            path,
            map_location=device,
        )


        # ====================================================
        # Model
        # ====================================================

        self.model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )
        
        self.unused_prototype_mask = (
            checkpoint.get(
                "unused_prototype_mask",
                None,
            )
        )


        # ====================================================
        # Thresholds
        # ====================================================

        self.recon_gate_threshold = (
            checkpoint.get(
                "recon_gate_threshold",
                None,
            )
        )


        self.static_gate_threshold = (
            checkpoint.get(
                "static_gate_threshold",
                None,
            )
        )


        # ====================================================
        # Optimizer
        # ====================================================

        if (
            load_optimizer
            and
            self.optimizer is not None
            and
            "optimizer" in checkpoint
        ):

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer"
                ]
            )


        # ====================================================
        # Scheduler
        # ====================================================

        if (
            load_scheduler
            and
            self.scheduler is not None
            and
            "scheduler" in checkpoint
        ):

            self.scheduler.load_state_dict(
                checkpoint[
                    "scheduler"
                ]
            )


        return checkpoint