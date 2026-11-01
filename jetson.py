import time
import csv
from pathlib import Path

import numpy as np
import torch

from utils.model_padding import PoseTTMVAD


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

WARMUP_ITERS = 10
LATENCY_ITERS = 50
THROUGHPUT_ITERS = 50

BATCH_SIZE = 1


EXPERIMENTS = [

    {
        "name": "W24_S4",
        "checkpoint":
            "/home/nrashvan/Token/results/ttm_best.pth.tar",

    },

    # {
    #     "name": "W72_S8",
    #     "checkpoint":
    #         "/home/nrashvan/Token/results/CHAD_s72_st36_t8_m64_d64/ttm_best.pth.tar",
    # },

]


OUTPUT_CSV = "ttm_edge_benchmark.csv"


# ============================================================
# Load model
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_config" not in checkpoint:
        raise KeyError(
            f"No model_config found in checkpoint:\n"
            f"{checkpoint_path}"
        )

    config = checkpoint[
        "model_config"
    ]


    # --------------------------------------------------------
    # Reconstruct model exactly as trained
    # --------------------------------------------------------

    model = PoseTTMVAD(

        num_keypoints=
            config["num_keypoints"],

        seg_len=
            config["seg_len"],
         
        
        
                    

        input_dim=
            config.get(
                "input_dim",
                2,
            ),

        embed_dim=
            config["embed_dim"],

        process_size=
            config["process_size"],

        memory_size=
            config["memory_size"],

        num_layers=
            config["num_layers"],

        mlp_dim=
            config["mlp_dim"],

        num_heads=
            config["num_heads"],

        dropout=
            config.get(
                "dropout",
                0.0,
            ),

        memory_mode=
            config.get(
                "memory_mode",
                "TL",
            ),

        use_positional_embedding=True,

        num_static_slots=
            config["num_static_slots"],

        window_num_layers=
            config.get(
                "window_num_layers",
                2,
            ),
    )


    # --------------------------------------------------------
    # Load learned weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )


    model = model.to(
        device
    )

    model.eval()


    return (
        model,
        config,
        checkpoint,
    )


# ============================================================
# Model forward

# ============================================================

def run_model(
    model,
    x,
    valid_mask,
    checkpoint,
):

    recon_threshold = checkpoint.get(
        "recon_gate_threshold",
        None,
    )

    static_threshold = checkpoint.get(
        "static_gate_threshold",
        None,
    )


    # --------------------------------------------------------
    # Use deployment/evaluation gating if available
    # --------------------------------------------------------

    if (
        recon_threshold is not None
        and
        static_threshold is not None
    ):

        return model(
            x,
            valid_mask=valid_mask,
            memory_update_mode="hard",
            recon_threshold=recon_threshold,
            static_threshold=static_threshold,
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return model(
        x,
        valid_mask=valid_mask,
        memory_update_mode="always",
    )


# ============================================================
# Benchmark one trained model
# ============================================================

def benchmark_model(
    model,
    checkpoint,
    num_steps,
    seg_len,
    input_dim,
    num_keypoints,
    device,
):

    # --------------------------------------------------------
    # Synthetic pose tensor
    #
    # [B,S,C,W,V]
    
    # --------------------------------------------------------

    x = torch.randn(
        BATCH_SIZE,
        num_steps,
        input_dim,
        seg_len,
        num_keypoints,
        device=device,
        dtype=torch.float32,
    )


    # No padding in synthetic benchmark
    valid_mask = torch.ones(
        BATCH_SIZE,
        num_steps,
        dtype=torch.bool,
        device=device,
    )


    # ========================================================
    # Warm-up
    # ========================================================

    print(
        f"Running {WARMUP_ITERS} warm-up iterations..."
    )


    with torch.inference_mode():

        for _ in range(
            WARMUP_ITERS
        ):

            _ = run_model(
                model,
                x,
                valid_mask,
                checkpoint,
            )


    torch.cuda.synchronize(
        device
    )


    # ========================================================
    # Latency
    #
    # Each measurement = one COMPLETE model forward.
  
    # ========================================================

    latency_ms = []


    print(
        f"Running {LATENCY_ITERS} latency iterations..."
    )


    with torch.inference_mode():

        for _ in range(
            LATENCY_ITERS
        ):

            torch.cuda.synchronize(
                device
            )

            start = time.perf_counter()


            _ = run_model(
                model,
                x,
                valid_mask,
                checkpoint,
            )


            torch.cuda.synchronize(
                device
            )

            end = time.perf_counter()


            elapsed_ms = (
                end
                -
                start
            ) * 1000.0


            latency_ms.append(
                elapsed_ms
            )


    mean_latency_ms = float(
        np.mean(
            latency_ms
        )
    )




    # ========================================================
    # Throughput
    #
    # Synchronize only at start and end.
    # ========================================================

    print(
        f"Running {THROUGHPUT_ITERS} "
        f"throughput iterations..."
    )


    with torch.inference_mode():

        torch.cuda.synchronize(
            device
        )

        start = time.perf_counter()


        for _ in range(
            THROUGHPUT_ITERS
        ):

            _ = run_model(
                model,
                x,
                valid_mask,
                checkpoint,
            )


        torch.cuda.synchronize(
            device
        )

        end = time.perf_counter()


    total_time = (
        end
        -
        start
    )


    # ========================================================
    # Throughput definitions
    # ========================================================

    forwards_per_second = (
        THROUGHPUT_ITERS
        *
        BATCH_SIZE
        /
        total_time
    )


    

    segments_per_second = (
        THROUGHPUT_ITERS
        *
        BATCH_SIZE
        *
        num_steps
        /
        total_time
    )


  


    return {

        "mean_latency_ms":
            mean_latency_ms,

        "forwards_per_second":
            forwards_per_second,


    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # CUDA check
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )


    device = torch.device(
        DEVICE
    )


    print()
    print("=" * 80)

    print(
        "GPU:",
        torch.cuda.get_device_name(
            device
        ),
    )

    print("=" * 80)


    all_results = []


    # ========================================================
    # Benchmark every trained model
    # ========================================================

    for experiment in EXPERIMENTS:

        name = experiment[
            "name"
        ]

        checkpoint_path = experiment[
            "checkpoint"
        ]

    


        print()
        print("=" * 80)

        print(
            f"MODEL: {name}"
        )

        print(
            f"Checkpoint: {checkpoint_path}"
        )

        print("=" * 80)


        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model, config, checkpoint = (
            load_model(
                checkpoint_path,
                device,
            )
        )


        # ====================================================
        # Read actual trained configuration
        # ====================================================

        seg_len = int(
            config[
                "seg_len"
            ]
        )
        
    

        input_dim = int(
            config.get(
                "input_dim",
                2,
            )
        )

        num_keypoints = int(
            config[
                "num_keypoints"
            ]
        )


        # ----------------------------------------------------
        # TTM steps
        # ----------------------------------------------------

        if (
            "ttm_sequence_length"
            in config
        ):

            num_steps = int(
                config[
                    "ttm_sequence_length"
                ]
            )

            step_source = (
                "checkpoint"
            )

        else:

            num_steps = int(
                experiment[
                    "ttm_steps_fallback"
                ]
            )

            step_source = (
                "fallback"
            )


    


        # ====================================================
        # Input shape
        # ====================================================

        input_shape = (
            BATCH_SIZE,
            num_steps,
            input_dim,
            seg_len,
            num_keypoints,
        )


        # ====================================================
        # Parameters
        # ====================================================

        num_parameters = sum(
            p.numel()
            for p in model.parameters()
        )


        trainable_parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )


        # ====================================================
        # Print configuration
        # ====================================================

        print()

        print(
            f"Segment length       : "
            f"{seg_len}"
        )

        

        print(
            f"TTM steps            : "
            f"{num_steps} "
            f"({step_source})"
        )

        # print(
        #     f"Temporal coverage    : "
        #     f"{temporal_coverage} frames"
        # )

        print(
            f"Input shape          : "
            f"{input_shape}"
        )

        print(
            f"Parameters           : "
            f"{num_parameters:,}"
        )

        print()


        # ====================================================
        # Benchmark
        # ====================================================

        result = benchmark_model(

            model=model,

            checkpoint=checkpoint,

            num_steps=num_steps,

            seg_len=seg_len,

            input_dim=input_dim,

            num_keypoints=num_keypoints,

            device=device,
        )


        # ====================================================
        # Store metadata
        # ====================================================

        result.update({

            "model":
                name,

            "checkpoint":
                checkpoint_path,

            "segment_length":
                seg_len,

            
                

            "ttm_steps":
                num_steps,

            # "temporal_coverage":
            #     temporal_coverage,

            "input_shape":
                str(input_shape),

            "parameters":
                num_parameters,

            "trainable_parameters":
                trainable_parameters,
        })


        all_results.append(
            result
        )


        # ====================================================
        # Individual result
        # ====================================================

        print()
        print("-" * 70)

        print(
            f"Mean latency / forward : "
            f"{result['mean_latency_ms']:.3f} "
            

        )

        print(
            f"Model forwards / sec   : "
            f"{result['forwards_per_second']:.2f}"
        )



        print("-" * 70)


        # ----------------------------------------------------
        # Release model before loading next one
        # ----------------------------------------------------

        del model

        torch.cuda.empty_cache()


    # ========================================================
    # Final table
    # ========================================================

    print()
    print("=" * 120)
    print("FINAL EDGE BENCHMARK")
    print("=" * 120)


    header = (
        f"{'Model':<14}"
        f"{'W':<7}"
        f"{'S':<7}"
        f"{'Latency(ms)':<15}"
        f"{'Forward/s':<13}"
    )

    print(
        header
    )

    print(
        "-" * 120
    )


    for r in all_results:

        compact_shape = (
            f"{r['ttm_steps']}"
            f"x"
            f"{r['segment_length']}"
            f"x18x2"
        )


        print(

            f"{r['model']:<14}"

            f"{r['segment_length']:<7}"

            

            f"{r['ttm_steps']:<7}"


            f"{r['mean_latency_ms']:<15.3f}"

            f"{r['forwards_per_second']:<13.2f}"

        
        )


    print(
        "=" * 120
    )


    # ========================================================
    # Save CSV
    # ========================================================

    if all_results:

        output_path = Path(
            OUTPUT_CSV
        )


        with open(
            output_path,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=
                    all_results[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                all_results
            )


        print(
            f"\nResults saved to: "
            f"{output_path.resolve()}"
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()