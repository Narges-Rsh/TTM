import torch
import torch.nn as nn

#loss for fixed-lenght +padding
class TTMVADLoss(nn.Module):

    def __init__(
        self,
        lambda_memory=0.05,
        lambda_static=0.1,
        # lambda_diversity=0.0,
    ):
        super().__init__()

        self.lambda_memory = (
            lambda_memory
        )

        self.lambda_static = (
            lambda_static
        )

        # self.lambda_diversity = (
        #     lambda_diversity
        # )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        target_pose,
        model_output,
        valid_mask,
    ):
        """
        Parameters
        ----------
        target_pose:

            [B,S,C,W,V]

        Example:

            [8,16,2,24,18]

        where:

            B = batch size

            S = fixed number of TTM steps
                e.g. 16

            C = pose coordinates
                x,y

            W = frames inside each TTM window
                e.g. 24

            V = number of keypoints
                e.g. 18


        valid_mask:

            [B,S]

        Example:

            Person/chunk 1:
                [1,1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,1]

            Person/chunk 2:
                [1,1,1,1,1,0,0,0,
                 0,0,0,0,0,0,0,0]
        """


        # ====================================================
        # Validate target pose
        # ====================================================

        if target_pose.ndim != 5:

            raise ValueError(
                "TTMVADLoss expects target_pose "
                "[B,S,C,W,V]. "
                f"Received {tuple(target_pose.shape)}"
            )


        B, S, C, W, V = (
            target_pose.shape
        )


        # ====================================================
        # Validate valid_mask
        # ====================================================

        if valid_mask.ndim != 2:

            raise ValueError(
                "valid_mask must have shape [B,S]. "
                f"Received {tuple(valid_mask.shape)}"
            )


        if tuple(
            valid_mask.shape
        ) != (B, S):

            raise ValueError(
                "valid_mask shape does not match "
                "target_pose. "
                f"Expected {(B, S)}, "
                f"received "
                f"{tuple(valid_mask.shape)}"
            )


        # ----------------------------------------------------
        # Make sure mask is on same device.
        # ----------------------------------------------------

        valid_mask = valid_mask.to(
            device=target_pose.device,
            dtype=torch.bool,
        )


        # Float version is useful for multiplication.
        #
        # True  -> 1.0
        # False -> 0.0

        valid_mask_float = valid_mask.to(
            dtype=target_pose.dtype
        )


        # ====================================================
        # Safety:
        #
        # Every chunk should contain at least one real window.
        # ====================================================

        if not torch.all(
            valid_mask.any(
                dim=1
            )
        ):

            raise ValueError(
                "Every TTM chunk must contain "
                "at least one valid window."
            )


        # ====================================================
        # 1. RECONSTRUCTION LOSS
        # ====================================================

        reconstruction = model_output[
            "reconstruction"
        ]


        if (
            reconstruction.shape
            !=
            target_pose.shape
        ):

            raise ValueError(
                "Reconstruction shape must match "
                "target_pose shape. "
                f"Target: {tuple(target_pose.shape)}, "
                f"reconstruction: "
                f"{tuple(reconstruction.shape)}"
            )


        # ----------------------------------------------------
        # Element-wise reconstruction squared error
        #
        # [B,S,C,W,V]
        # ----------------------------------------------------

        reconstruction_error = (

            target_pose
            -
            reconstruction

        ).pow(2)


        # ----------------------------------------------------
        # Reconstruction error for EACH TTM step.
        #
        # Average over:
        #
        # C = coordinates
        # W = frames
        # V = joints
        #
        # [B,S,C,W,V]
        #
        # ->
        #
        # [B,S]
        # ----------------------------------------------------

        recon_per_step = (

            reconstruction_error
            .mean(
                dim=(
                    2,
                    3,
                    4,
                )
            )
        )


        # ----------------------------------------------------
        # Remove padded positions.
        #
        # Example:
        #
        # recon:
        # [0.2,0.3,0.1,0.4,...]
        #
        # mask:
        # [1,1,1,0,...]
        #
        # result:
        # [0.2,0.3,0.1,0,...]
        # ----------------------------------------------------

        masked_recon_per_step = (

            recon_per_step

            *

            valid_mask_float
        )


        # ----------------------------------------------------
        # Number of REAL windows in each batch sample.
        #
        # [B]
        #
        # Example:
        #
        # [16, 7, 12, 3, ...]
        # ----------------------------------------------------

        valid_count_per_sample = (

            valid_mask_float
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )


        # ----------------------------------------------------
        # Average reconstruction error PER CHUNK.
        #
        # [B]
        # ----------------------------------------------------

        recon_per_sample = (

            masked_recon_per_step
            .sum(
                dim=1
            )

            /

            valid_count_per_sample
        )


        # ----------------------------------------------------
        # Global reconstruction training loss.
        #
        # Every REAL window receives equal weight.
        #
        # Padding receives zero weight.
        # ----------------------------------------------------

        # total_valid_windows = (

        #     valid_mask_float
        #     .sum()
        #     .clamp_min(
        #         1.0
        #     )
        # )


        # recon_loss = (

        #     masked_recon_per_step
        #     .sum()

        #     /

        #     total_valid_windows
        # )
        
        recon_loss = (
            recon_per_sample.mean()
        )


        # ====================================================
        # 2. DYNAMIC-MEMORY LOSS
        # ====================================================

        memory_shift_per_step = model_output[
            "memory_shift_per_step"
        ]


        if (
            memory_shift_per_step.ndim
            !=
            2
        ):

            raise ValueError(
                "memory_shift_per_step must "
                "have shape [B,S]. "
                f"Received "
                f"{tuple(memory_shift_per_step.shape)}"
            )


        if tuple(
            memory_shift_per_step.shape
        ) != (B, S):

            raise ValueError(
                "memory_shift_per_step must "
                "have shape [B,S]. "
                f"Expected {(B, S)}, "
                f"received "
                f"{tuple(memory_shift_per_step.shape)}"
            )


        # ----------------------------------------------------
        # Memory sequence inside ONE chunk:
        #
        # M0 -> M1 -> M2 -> ... -> MS
        #
        # The FIRST transition:
        #
        # M0 -> M1
        #
        # should NOT be regularized because M0 is newly
        # initialized memory.
        #
        # We regularize:
        #
        # M1 -> M2
        # M2 -> M3
        # ...
        #
        # Therefore:
        #
        # valid mask:
        #
        # 1 1 1 1 1 0 0
        #
        # memory-valid mask:
        #
        # 0 1 1 1 1 0 0
        # ----------------------------------------------------

        memory_valid_mask = (
            valid_mask.clone()
        )


        if S > 0:

            memory_valid_mask[
                :,
                0
            ] = False


        memory_mask_float = (

            memory_valid_mask.to(
                dtype=target_pose.dtype
            )
        )


        # ----------------------------------------------------
        # Remove:
        #
        # - first step of every chunk
        # - padded positions
        # ----------------------------------------------------

        masked_memory_per_step = (

            memory_shift_per_step

            *

            memory_mask_float
        )


        # ----------------------------------------------------
        # Number of meaningful memory transitions
        # per sample.
        #
        # [B]
        # ----------------------------------------------------

        memory_count_per_sample = (

            memory_mask_float
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )


        # ----------------------------------------------------
        # Dynamic memory loss PER SAMPLE.
        #
        # [B]
        # ----------------------------------------------------

        memory_per_sample = (

            masked_memory_per_step
            .sum(
                dim=1
            )

            /

            memory_count_per_sample
        )


        # ----------------------------------------------------
        # Global dynamic-memory training loss.
        #
        # Average over all REAL, meaningful
        # memory transitions.
        # ----------------------------------------------------

        # total_memory_transitions = (

        #     memory_mask_float
        #     .sum()
        # )


        # memory_loss = (

        #     masked_memory_per_step
        #     .sum()

        #     /

        #     total_memory_transitions
        #     .clamp_min(
        #         1.0
        #     )
        # )
        memory_loss = (
            memory_per_sample.mean()
        )

        # ====================================================
        # 3. STATIC NORMAL-MEMORY LOSS
        # ====================================================

        static_distance = model_output[
            "static_distance"
        ]


        if (
            static_distance.ndim
            !=
            2
        ):

            raise ValueError(
                "static_distance must have "
                "shape [B,S]. "
                f"Received "
                f"{tuple(static_distance.shape)}"
            )


        if tuple(
            static_distance.shape
        ) != (B, S):

            raise ValueError(
                "static_distance must have "
                "shape [B,S]. "
                f"Expected {(B, S)}, "
                f"received "
                f"{tuple(static_distance.shape)}"
            )


        # ----------------------------------------------------
        # Remove padded positions.
        # ----------------------------------------------------

        masked_static_per_step = (

            static_distance

            *

            valid_mask_float
        )


        # ----------------------------------------------------
        # Static-memory loss PER SAMPLE.
        #
        # [B]
        # ----------------------------------------------------

        static_per_sample = (

            masked_static_per_step
            .sum(
                dim=1
            )

            /

            valid_count_per_sample
        )


        # ----------------------------------------------------
        # Global static-memory training loss.
        #
        # Average over all REAL windows.
        # ----------------------------------------------------

        # static_loss = (

        #     masked_static_per_step
        #     .sum()

        #     /

        #     total_valid_windows
        # )
        
        static_loss = (
            static_per_sample.mean()
        )


        # ====================================================
        # # 4. STATIC PROTOTYPE DIVERSITY
        # # ====================================================

        # static_diversity = model_output[
        #     "static_diversity"
        # ]


        # if static_diversity.ndim != 0:

        #     raise ValueError(
        #         "static_diversity must be "
        #         "a scalar tensor. "
        #         f"Received shape "
        #         f"{tuple(static_diversity.shape)}"
        #     )


        # ====================================================
        # 5. TOTAL LOSS PER SAMPLE
        #
        # Mainly useful for diagnostics.
        #
        # Diversity is NOT included here because diversity
        # belongs to the entire global prototype bank rather
        # than one particular sample.
        # ====================================================

        total_per_sample = (

            recon_per_sample

            +

            self.lambda_memory
            *
            memory_per_sample

            +

            self.lambda_static
            *
            static_per_sample
        )


        # ====================================================
        # 6. FINAL GLOBAL TRAINING LOSS
        #
        # L =
        #
        # L_reconstruction
        #
        # +
        #
        # lambda_memory
        # *
        # L_dynamic_memory
        #
        # +
        #
        # lambda_static
        # *
        # L_static_memory
        #
        # +
        #
        # lambda_diversity
        # *
        # L_static_diversity
        # ====================================================

        total_loss = (

            total_per_sample.mean()
            
        )


        # ====================================================
        # Return
        # ====================================================

        return {

            # ------------------------------------------------
            # Scalar used for backward()
            # ------------------------------------------------

            "loss":
                total_loss,


            # ------------------------------------------------
            # Total loss per chunk/sample
            #
            # [B]
            # ------------------------------------------------

            "total_per_sample":
                total_per_sample,


            # =================================================
            # Reconstruction
            # =================================================

            # [B]
            "reconstruction":
                recon_per_sample,


            # [B,S]
            #
            # Padding positions are zero.
            "reconstruction_per_step":
                masked_recon_per_step,


            # Scalar global reconstruction loss
            "reconstruction_loss":
                recon_loss,


            # =================================================
            # Dynamic memory
            # =================================================

            # [B]
            "memory_shift":
                memory_per_sample,


            # [B,S]
            #
            # First position and padding positions are zero.
            "memory_shift_per_step":
                masked_memory_per_step,


            # Scalar global memory loss
            "memory_loss":
                memory_loss,


            # =================================================
            # Static memory
            # =================================================

            # [B]
            "static_memory":
                static_per_sample,


            # [B,S]
            #
            # Padding positions are zero.
            "static_memory_per_step":
                masked_static_per_step,


            # Scalar global static loss
            "static_loss":
                static_loss,


            # =================================================
            # Static-memory diversity
            # =================================================

            # "static_diversity":
            #     static_diversity,


            # =================================================
            # Masks / counts useful for debugging
            # =================================================

            "valid_mask":
                valid_mask,

            "memory_valid_mask":
                memory_valid_mask,

            "num_valid_windows":
                valid_mask.sum(),

            "num_valid_memory_transitions":
                memory_valid_mask.sum(),
        }