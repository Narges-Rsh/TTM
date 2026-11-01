import torch
import torch.nn as nn
import torch.nn.functional as F


from token_turing import TokenTuringMachineEncoder

#fixed-length, padded model
# ============================================================
# Spatiotemporal pose-window encoder
# ============================================================

class PoseWindowEncoder(nn.Module):
    """
    Encode ONE complete pose window.

    Input:
        [B, W, V, C]

    Output:
        [B, W*V, D]

    Each frame-joint location becomes one spatiotemporal token.
    """

    def __init__(
        self,
        seg_len,
        num_keypoints,
        input_dim=2,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        dropout=0.0,
    ):
        super().__init__()

        self.seg_len = seg_len
        self.num_keypoints = num_keypoints

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # ====================================================
        # Project raw x,y coordinates into embedding space
        # ====================================================

        self.projection = nn.Linear(
            input_dim,
            embed_dim,
        )

        self.joint_embedding = nn.Parameter(
            torch.zeros(
                1,
                1,
                num_keypoints,
                embed_dim,
            )
        )


        self.temporal_embedding = nn.Parameter(
            torch.zeros(
                1,
                seg_len,
                1,
                embed_dim,
            )
        )

        nn.init.normal_(
            self.joint_embedding,
            std=0.02,
        )

        nn.init.normal_(
            self.temporal_embedding,
            std=0.02,
        )

        # ====================================================
        # Input normalization
        # ====================================================

        self.input_norm = nn.LayerNorm(
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # ====================================================
        # Spatiotemporal Transformer
        #
        # After flattening:
        #
        # [B,W,V,D]
        #
        # ->
        #
        # [B,W*V,D]

        # ====================================================

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=
                embed_dim,

            nhead=
                num_heads,

            dim_feedforward=
                mlp_dim,

            dropout=
                dropout,

            activation=
                "gelu",

            batch_first=
                True,

            norm_first=
                True,
        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=
                num_layers,
        )

        self.output_norm = nn.LayerNorm(
            embed_dim
        )


    def forward(
        self,
        x,
    ):
        """
        x:
            [B,W,V,C]

        returns:
            [B,W*V,D]
        """

        if x.ndim != 4:

            raise ValueError(
                "PoseWindowEncoder expects "
                "[B,W,V,C]. "
                f"Received {tuple(x.shape)}"
            )

        B, W, V, C = x.shape

        if W != self.seg_len:

            raise ValueError(
                f"Expected window length "
                f"{self.seg_len}, "
                f"received {W}."
            )

        if V != self.num_keypoints:

            raise ValueError(
                f"Expected "
                f"{self.num_keypoints} keypoints, "
                f"received {V}."
            )

        if C != self.input_dim:

            raise ValueError(
                f"Expected {self.input_dim} "
                f"input channels, "
                f"received {C}."
            )


        x = self.projection(
            x
        )

        # ====================================================
        # Add spatial + temporal identity
        # ====================================================

        x = (

            x

            +

            self.joint_embedding

            +

            self.temporal_embedding
        )

        x = self.input_norm(
            x
        )

        x = self.dropout(
            x
        )

        x = x.reshape(
            B,
            W * V,
            self.embed_dim,
        )


        x = self.transformer(
            x
        )

        x = self.output_norm(
            x
        )

        return x


# ============================================================
# Reconstruction head
# ============================================================

class PoseWindowReconstructionHead(nn.Module):
    """
    Reconstruct ONE complete 24-frame pose window
    from the TTM controller output.

    Input:
        [B,R,D]

    Output:
        [B,W,V,C]

    Example:
        [B,24,18,2]
    """

    def __init__(
        self,
        embed_dim,
        process_size,
        seg_len,
        num_keypoints,
        hidden_dim=512,
        output_dim=2,
    ):
        super().__init__()

        self.seg_len = seg_len
        self.num_keypoints = num_keypoints
        self.output_dim = output_dim

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                process_size
                *
                embed_dim,

                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(

                hidden_dim,

                seg_len
                *
                num_keypoints
                *
                output_dim,
            ),
        )


    def forward(
        self,
        x,
    ):
        """
        x:
            [B,R,D]

        output:
            [B,W,V,C]
        """

        B = x.shape[0]

        x = self.norm(
            x
        )

        x = x.reshape(
            B,
            -1,
        )

        x = self.decoder(
            x
        )

        x = x.reshape(

            B,

            self.seg_len,

            self.num_keypoints,

            self.output_dim,
        )

        return x


# ============================================================
# Static normal memory
# ============================================================

class StaticNormalMemoryBank(nn.Module):
    """
    Static normal-memory bank.

    Training:
        1. Warm-up:
           The first K valid normal states directly initialize
           the K prototype slots.

        2. After warm-up:
           The current state is compared with all prototypes
           using cosine similarity.

           Only the most similar prototype is updated:

               P_k <- 0.5 * P_k + 0.5 * state

    Inference:
        Prototypes remain fixed.
    """

    def __init__(
        self,
        num_slots,
        embed_dim,
        
    ):
        super().__init__()

        self.num_slots = num_slots
        self.embed_dim = embed_dim
        

        # --------------------------------------------------------
        # Static memory itself
        #
        # Not an nn.Parameter anymore.
        # It is explicitly written during training.
        # --------------------------------------------------------

        self.register_buffer(
            "prototypes",
            torch.zeros(
                num_slots,
                embed_dim,
            )
        )

        # Number of prototype slots initialized during warm-up
        self.register_buffer(
            "num_initialized",
            torch.zeros(
                (),
                dtype=torch.long,
            )
        )


    def forward(
        self,
        x,
    ):
        """
        READ from static memory.

        x:
            [B,D]

        Returns:
            distance:
                [B]

            best_similarity:
                [B]

            assignment:
                [B]
        """

        B = x.shape[0]

        # ========================================================
        # During warm-up, static-memory anomaly distance is not
        # meaningful yet because the bank is incomplete.
        # ========================================================

        if self.num_initialized.item() < self.num_slots:

            distance = torch.zeros(
                B,
                device=x.device,
                dtype=x.dtype,
            )

            best_similarity = torch.zeros(
                B,
                device=x.device,
                dtype=x.dtype,
            )

            assignment = torch.full(
                (B,),
                -1,
                device=x.device,
                dtype=torch.long,
            )

            return (
                distance,
                best_similarity,
                assignment,
            )

        # ========================================================
        # Bank is full: calculate cosine similarity
        # ========================================================

        x_normalized = F.normalize(
            x,
            dim=-1,
        )

        # Clone is important because prototypes will be explicitly
        # updated later in the same forward pass.
        prototype_snapshot = (
            self.prototypes
            .detach()
            .clone()
        )

        prototypes_normalized = F.normalize(
            prototype_snapshot,
            dim=-1,
        )

        # [B,K]
        similarity_all = torch.matmul(
            x_normalized,
            prototypes_normalized.t(),
        )

        # [B]
        best_similarity, assignment = (
            similarity_all.max(
                dim=-1
            )
        )

        # cosine distance
        distance = (
            1.0
            -
            best_similarity
        )

        return (
            distance,
            best_similarity,
            assignment,
        )


    @torch.no_grad()
    def write(
        self,
        x,
        valid_mask=None,
    ):
        """
        WRITE into static memory.

        x:
            [B,D]

        During warm-up:
            directly fill P1, P2, ..., PK.

        After warm-up:
            find the prototype with maximum cosine similarity
            and perform:

                P_k <- 0.5 * P_k + 0.5 * x
        """

        x = x.detach()

        B = x.shape[0]

        if valid_mask is None:

            valid_mask = torch.ones(
                B,
                dtype=torch.bool,
                device=x.device,
            )

        for b in range(B):

            if not valid_mask[b]:
                continue

            state = x[b]

            # ====================================================
            # WARM-UP
            # ====================================================

            if self.num_initialized.item() < self.num_slots:

                slot = int(
                    self.num_initialized.item()
                )

                self.prototypes[
                    slot
                ].copy_(
                    state
                )

                self.num_initialized += 1

                continue

            # ====================================================
            # AFTER WARM-UP
            #
            # Find most similar prototype
            # ====================================================

            state_normalized = F.normalize(
                state,
                dim=-1,
            )

            prototypes_normalized = F.normalize(
                self.prototypes,
                dim=-1,
            )

            # [K]
            similarity = torch.matmul(
                prototypes_normalized,
                state_normalized,
            )

            best_slot = int(
                similarity.argmax().item()
            )

            # ====================================================
            # 50-50 UPDATE
            #
            # P_k(new) =
            #     0.5 * P_k(old)
            #     +
            #     0.5 * state
            # ====================================================

            updated_prototype = (
                0.5
                *
                self.prototypes[
                    best_slot
                ]
                +
                0.5
                *
                state
            )

            self.prototypes[
                best_slot
            ].copy_(
                updated_prototype
            )


# ============================================================
# Dual-memory Pose TTM
# ============================================================

class PoseTTMVAD(nn.Module):

    def __init__(
        self,
        num_keypoints,

        # IMPORTANT:
        #
        # Number of frames INSIDE each TTM step/window.
        seg_len=24,

        input_dim=2,
        embed_dim=64,

        process_size=8,
        memory_size=32,

        num_layers=2,
        mlp_dim=128,
        num_heads=4,

        dropout=0.0,

        memory_mode="TL",
        use_positional_embedding=True,

        # Static normal memory
        num_static_slots=16,
        

        # Number of Transformer layers used to encode
        # the w-frame pose window.
        window_num_layers=2,
    ):
        super().__init__()

        self.num_keypoints = (
            num_keypoints
        )

        self.seg_len = (
            seg_len
        )

        self.input_dim = (
            input_dim
        )

        self.embed_dim = (
            embed_dim
        )

        self.process_size = (
            process_size
        )

        self.memory_size = (
            memory_size
        )

        # ====================================================
        # 1. Spatiotemporal pose-window encoder
        #
        # One 24-frame window:
        #
        # [B,24,18,2]
        #
        # ->
        #
        # [B,432,D]
        # ====================================================

        self.pose_encoder = PoseWindowEncoder(

            seg_len=
                seg_len,

            num_keypoints=
                num_keypoints,

            input_dim=
                input_dim,

            embed_dim=
                embed_dim,

            num_layers=
                window_num_layers,

            num_heads=
                num_heads,

            mlp_dim=
                mlp_dim,

            dropout=
                dropout,
        )

        # ====================================================
        # 2. Dynamic memory = TTM
        #
        # Input at ONE TTM step:
        #
        # W × V spatiotemporal tokens
        # TTM read operation then compresses:
        #
        # [dynamic memory + current-window tokens]
        #
        # into process_size tokens.
        # ====================================================

        self.ttm = TokenTuringMachineEncoder(

            dim=
                embed_dim,

            process_size=
                process_size,

            memory_size=
                memory_size,

            memory_mode=
                memory_mode,

            processing_unit=
                "transformer",

            num_layers=
                num_layers,

            mlp_dim=
                mlp_dim,

            num_heads=
                num_heads,

            dropout=
                dropout,

            use_positional_embedding=
                use_positional_embedding,

            max_input_tokens=
                seg_len * num_keypoints
        )
        

        # ====================================================
        # 3. Reconstruct complete window
        # ====================================================

        self.reconstruction_head = (
            PoseWindowReconstructionHead(

                embed_dim=
                    embed_dim,

                process_size=
                    process_size,

                seg_len=
                    seg_len,

                num_keypoints=
                    num_keypoints,

                hidden_dim=
                    512,

                output_dim=
                    input_dim,
            )
        )

        # ====================================================
        # 4. State representation for static normal memory
        # ====================================================

        self.state_norm = nn.LayerNorm(
            embed_dim
        )

        # ====================================================
        # 5. Static normal memory
        # ====================================================

        self.static_memory = (
            StaticNormalMemoryBank(

                num_slots=
                    num_static_slots,

                embed_dim=
                    embed_dim,

                # diversity_margin=
                #     static_diversity_margin,
            )
        )


    # ========================================================
    # Dynamic memory initialization
    # ========================================================

    def init_dynamic_memory(
        self,
        batch_size,
        device,
        dtype,
    ):

        return self.ttm.init_memory(

            batch_size=
                batch_size,

            device=
                device,

            dtype=
                dtype,
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        pose,

        initial_memory=None,
        valid_mask=None,
        memory_update_mode="always",

        recon_threshold=None,
        static_threshold=None,
    ):
        """
        INPUT
        =====

        pose:

            [B,S,C,W,V]

        Current experiment:

            B = 1

            S = fixed number of TTM steps
        e.g. 16

            C = 2
                x,y

            W = 24
                frames inside one TTM step

            V = 18
                joints


          valid_mask:
        [B,S]

        True  = real 24-frame window
        False = padded position

Each chunk starts from newly initialized
dynamic memory unless initial_memory is supplied.
Initialize dynamic memory ONCE FOR EACH CHUNK
# Memory after the final step of this chunk.

        TTM TEMPORAL PROCESSING
        =======================

        Window 1:
            [B,2,24,18]
                ↓
            432 spatiotemporal tokens
                ↓
            TTM step 1
                ↓
            M1

        Window 2:
                ↓
            TTM step 2
                ↓
            M2

        ...

        Window S:
                ↓
            MS


        OUTPUTS
        =======

        reconstruction:
            [B,S,2,W,V]

        memories:
            [B,S,M,D]

        candidate_memories:
            [B,S,M,D]

        ttm_outputs:
            [B,S,R,D]

        states:
            [B,S,D]

        static_distance:
            [B,S]

        reconstruction_per_step:
            [B,S]

        memory_shift_per_step:
            [B,S]

        final_memory:
            [B,M,D]
        """

        # ====================================================
        # Validate input
        # ====================================================

        if pose.ndim != 5:

            raise ValueError(
                "Expected pose "
                "[B,S,C,W,V]. "
                f"Received {tuple(pose.shape)}"
            )

        B, S, C, W, V = (
            pose.shape
        )

        if C != self.input_dim:

            raise ValueError(
                f"Expected "
                f"{self.input_dim} channels, "
                f"received {C}."
            )

        if W != self.seg_len:

            raise ValueError(
                f"Expected window length "
                f"{self.seg_len}, "
                f"received {W}."
            )

        if V != self.num_keypoints:

            raise ValueError(
                f"Expected "
                f"{self.num_keypoints} keypoints, "
                f"received {V}."
            )

        if S < 1:

            raise ValueError(
                "A person sequence must contain "
                "at least one TTM step."
            )
            
                # ========================================================
        # Valid TTM-step mask
        #
        # valid_mask:
        #     [B,S]
        # ========================================================

        if valid_mask is None:

            # Backward compatibility:
            # if no mask is supplied, assume every step is real.

            valid_mask = torch.ones(

                B,
                S,

                dtype=torch.bool,

                device=pose.device,
            )

        else:

            if valid_mask.ndim != 2:

                raise ValueError(
                    "valid_mask must have shape [B,S]. "
                    f"Received {tuple(valid_mask.shape)}"
                )

            if tuple(valid_mask.shape) != (B, S):

                raise ValueError(
                    "valid_mask shape does not match pose. "
                    f"Expected {(B, S)}, "
                    f"received {tuple(valid_mask.shape)}"
                )

            valid_mask = valid_mask.to(
                device=pose.device,
                dtype=torch.bool,
            )


        # Every chunk created by our dataset should contain
        # at least one real window.

        if not torch.all(
            valid_mask.any(dim=1)
        ):

            raise ValueError(
                "Every TTM chunk must contain at least "
                "one valid window."
            )
            
       # Padding must be right-padded only.
        #
        # Valid:
        # 1 1 1 1 0 0 0
        #
        # Invalid:
        # 1 1 0 1 0 0 0

        if S > 1:

            invalid_transition = (
                (~valid_mask[:, :-1])
                &
                valid_mask[:, 1:]
            )

            if torch.any(invalid_transition):

                raise ValueError(
                    "valid_mask must contain all real windows "
                    "before padded positions."
                )     
                
                
            

        # ========================================================
    # Encode ALL windows in ONE encoder call
    # ========================================================
    #
    # Input pose:
    #
    #     [B,S,C,W,V]
    # ========================================================


    # --------------------------------------------------------
    # Move coordinate channel C to the last dimension
    #
    # [B,S,C,W,V]
    #
    # ->
    #
    # [B,S,W,V,C]
    # --------------------------------------------------------

        pose_windows = pose.permute(
            0,
            1,
            3,
            4,
            2,
        ).contiguous()


    # --------------------------------------------------------
    # Merge B and S so every window becomes one batch item
    #
    # [B,S,W,V,C]
    #
    # ->
    #
    # [B*S,W,V,C]
    # --------------------------------------------------------

        flat_windows = pose_windows.reshape(
            B * S,
            W,
            V,
            C,
        )


    # --------------------------------------------------------
    # ONE call to PoseWindowEncoder
    #
    # [B*S,W,V,C]
    #
    # ->
    #
    # [B*S,W*V,D]
    # --------------------------------------------------------

        flat_window_tokens = self.pose_encoder(
            flat_windows
        )


    # --------------------------------------------------------
    # Restore B and S dimensions
    #
    # [B*S,W*V,D]
    #
    # ->
    #
    # [B,S,W*V,D]
    # --------------------------------------------------------

        all_window_tokens = flat_window_tokens.reshape(
            B,
            S,
            W * V,
            self.embed_dim,
        )

        

        # ====================================================
        # Initialize dynamic memory ONCE FOR THIS PERSON
        # ====================================================

        if initial_memory is None:

            memory = (
                self.init_dynamic_memory(

                    batch_size=
                            B,

                    device=
                            pose.device,

                    dtype=
                            pose.dtype,
                    )
                )

        else:

            memory = (
                    initial_memory
            )

        # ====================================================
        # Storage
        # ====================================================

        reconstruction_list = []

        memory_list = []
        candidate_memory_list = []

        output_list = []
        state_list = []

        static_distance_list = []
        static_assignment_list = []

        reconstruction_error_list = []
        memory_shift_list = []

        gate_list = []

        # ====================================================
        # IMPORTANT:
        #
        # Loop over S WINDOWS.
      
        # ====================================================

        for s in range(S):
            
            # =====================================================
            # Is this TTM step real or padding?
            # =====================================================

            step_valid_bool = valid_mask[
                :,
                s,
            ]

            # [B]

            step_valid = step_valid_bool.to(
                dtype=pose.dtype
            )

            # [B]
            
            
            # ------------------------------------------------
    # Current original pose window
            #
            # pose_windows:
            #     [B,S,W,V,C]
            #
            # current_window_xy:
            #     [B,W,V,C]
            # ------------------------------------------------

            current_window_xy = pose_windows[
                :,
                s,
            ]


            # ------------------------------------------------
            # Current ALREADY-ENCODED window
            #
            # all_window_tokens:
            #     [B,S,W*V,D]
            #
            # window_tokens:
            #     [B,W*V,D]
            # ------------------------------------------------

            window_tokens = all_window_tokens[
                :,
                s,
            ]


            previous_memory = (
                memory
            )
                    


            (
                candidate_memory,
                output_tokens,

            ) = self.ttm.forward_step(

                input_tokens=
                    window_tokens,

                memory_tokens=
                    previous_memory,
            )

            # =================================================
            # Reconstruct entire 24-frame window
            #
            # [B,R,D]
            #
            # ->
            #
            # [B,W,V,2]
            # =================================================

            recon_window = (
                self.reconstruction_head(

                    output_tokens
                )
            )

            # =================================================
            # Reconstruction anomaly signal
            #
            # Average over:
            #
            # W, V, C
            #
            # ->
            #
            # [B]
            # =================================================

            reconstruction_error_raw = (

                (
                    current_window_xy
                    -
                    recon_window
                )
                .pow(2)
                .mean(
                    dim=(1, 2, 3)
                )
            )

            # =================================================
            # Dynamic-memory shift
            #
            # candidate memory:
            #
            # [B,M,D]
            #
            # previous memory:
            #
            # [B,M,D]
            #
            # ->
            #
            # [B]
            # =================================================

            memory_shift_raw = (

                (
                    candidate_memory
                    -
                    previous_memory
                )
                .pow(2)
                .mean(
                    dim=(1, 2)
                )
            )

            # =================================================
            # Current temporal state
            #
            # output_tokens:
            #
            # [B,R,D]
            #
            # mean over R:
            #
            # [B,D]
            # =================================================

            state = output_tokens.mean(
                dim=1
            )

            state = self.state_norm(
                state
            )

            # =================================================
            # Compare state with static normal memory
            # =================================================

            (
                static_distance_raw,
                _,
                static_assignment,

            ) = self.static_memory(
                state
            )
            
            
            # =================================================
            # WRITE static normal memory
            #
            # TRAINING ONLY
            # =================================================

            if self.training:

                self.static_memory.write(
                    state,
                    valid_mask=step_valid_bool,
                )

            # =====================================================
            # Dynamic-memory gating
            # =====================================================

            if (
                memory_update_mode
                ==
                "always"
            ):

                # -------------------------------------------------
                # Training:
                #
                # Real windows:
                #     update memory
                #
                # Padding:
                #     do NOT update memory
                # -------------------------------------------------

                gate = step_valid


            elif (
                memory_update_mode
                ==
                "hard"
            ):

                if recon_threshold is None:

                    raise ValueError(
                        "recon_threshold is required "
                        "for hard memory gating."
                    )

                if static_threshold is None:

                    raise ValueError(
                        "static_threshold is required "
                        "for hard memory gating."
                    )


                normal_recon = (

                    reconstruction_error_raw
                    .detach()

                    <=

                    recon_threshold
                )


                normal_static = (

                    static_distance_raw
                    .detach()

                    <=

                    static_threshold
                )


                # -------------------------------------------------
                # Memory can update ONLY when:
                #
                # 1. this is a REAL window
                # 2. reconstruction looks normal
                # 3. static state looks normal
                # -------------------------------------------------

                gate = (

                    step_valid_bool

                    &

                    normal_recon

                    &

                    normal_static

                ).to(
                    dtype=pose.dtype
                )


            else:

                raise ValueError(
                    "Unknown memory_update_mode: "
                    f"{memory_update_mode}"
                )
        
        
        
        
        

            # =================================================
            # Gated memory update
            # =================================================

            gate_3d = gate.view(
                B,
                1,
                1,
            )

            memory = (

                gate_3d
                *
                candidate_memory

                +

                (
                    1.0
                    -
                    gate_3d
                )
                *
                previous_memory
            )
            
            
            # =====================================================
            # Remove padded steps from anomaly signals
            # =====================================================

            reconstruction_error = (

                reconstruction_error_raw
                *
                step_valid
            )

            memory_shift = (

                memory_shift_raw
                *
                step_valid
            )

            static_distance = (

                static_distance_raw
                *
                step_valid
            )


            # Padding has no meaningful prototype assignment.

            static_assignment = torch.where(

                step_valid_bool,

                static_assignment,

                torch.full_like(
                    static_assignment,
                    -1,
                ),
            )
            
            

            # =================================================
            # Store current TTM-step outputs
            # =================================================

            reconstruction_list.append(
                recon_window
            )

            memory_list.append(
                memory
            )

            candidate_memory_list.append(
                candidate_memory
            )

            output_list.append(
                output_tokens
            )

            state_list.append(
                state
            )

            static_distance_list.append(
                static_distance
            )

            static_assignment_list.append(
                static_assignment
            )

            reconstruction_error_list.append(
                reconstruction_error
            )

            memory_shift_list.append(
                memory_shift
            )

            gate_list.append(
                gate
            )

        # ====================================================
        # Stack all S TTM steps
        # ====================================================

        # ----------------------------------------------------
        # Reconstruction
        #
        # list of:
        # [B,W,V,C]
        #
        # ->
        #
        # [B,S,W,V,C]
        # ----------------------------------------------------

        reconstruction = torch.stack(

            reconstruction_list,

            dim=1,
        )

        # ----------------------------------------------------
        # Convert to trainer format:
        #
        # [B,S,W,V,C]
        #
        # ->
        #
        # [B,S,C,W,V]
        # ----------------------------------------------------

        reconstruction = (
            reconstruction.permute(

                0,
                1,
                4,
                2,
                3,

            ).contiguous()
        )

        # [B,S,C,W,V]


        # ====================================================
        # Dynamic memories
        # ====================================================

        memories = torch.stack(

            memory_list,

            dim=1,
        )

        # [B,S,M,D]


        candidate_memories = torch.stack(

            candidate_memory_list,

            dim=1,
        )

        # [B,S,M,D]


        # ====================================================
        # TTM controller outputs
        # ====================================================

        ttm_outputs = torch.stack(

            output_list,

            dim=1,
        )

        # [B,S,R,D]


        # ====================================================
        # Temporal states
        # ====================================================

        states = torch.stack(

            state_list,

            dim=1,
        )

        # [B,S,D]


        # ====================================================
        # Static normal-memory signals
        # ====================================================

        static_distance = torch.stack(

            static_distance_list,

            dim=1,
        )

        # [B,S]


        static_assignment = torch.stack(

            static_assignment_list,

            dim=1,
        )

        # [B,S]


        # ====================================================
        # Per-TTM-step reconstruction score
        # ====================================================

        reconstruction_per_step = torch.stack(

            reconstruction_error_list,

            dim=1,
        )

        # [B,S]


        # ====================================================
        # Per-TTM-step dynamic-memory shift
        # ====================================================

        memory_shift_per_step = torch.stack(

            memory_shift_list,

            dim=1,
        )

        # [B,S]


        # ====================================================
        # Memory-update gate
        # ====================================================

        memory_update_gate = torch.stack(

            gate_list,

            dim=1,
        )     

        # ====================================================
        # Return
        # ====================================================

        return {

            # ------------------------------------------------
            # Reconstruction
            #
            # [B,S,2,W,V]
            # ------------------------------------------------

            "reconstruction":
                reconstruction,


            # ------------------------------------------------
            # Dynamic memory
            #
            # [B,S,M,D]
            # ------------------------------------------------

            "memories":
                memories,


            # ------------------------------------------------
            # Candidate dynamic memory before gate
            #
            # [B,S,M,D]
            # ------------------------------------------------

            "candidate_memories":
                candidate_memories,


            # ------------------------------------------------
            # Controller output
            #
            # [B,S,R,D]
            # ------------------------------------------------

            "ttm_outputs":
                ttm_outputs,


            # ------------------------------------------------
            # Compact temporal state
            #
            # [B,S,D]
            # ------------------------------------------------

            "states":
                states,


            # ------------------------------------------------
            # Static-normal-memory information
            #
            # [B,S]
            # ------------------------------------------------

            "static_distance":
                static_distance,

            "static_assignment":
                static_assignment,

            "reconstruction_per_step":
                reconstruction_per_step,

            "memory_shift_per_step":
                memory_shift_per_step,



            "memory_update_gate":
                memory_update_gate,


            

            "final_memory":
                memory,
                
            "valid_mask":
                valid_mask,
        }