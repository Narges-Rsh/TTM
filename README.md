# Token Turing Machine — PyTorch

A clean standalone PyTorch port of the core Token Turing Machine (TTM)
architecture from Google Research Scenic.

## Project

```text
ttm_pytorch/
├── token_turing/
│   ├── __init__.py
│   └── model.py
├── test_ttm.py
└── requirements.txt
```

## Conda setup

CPU / generic:

```bash
conda create -n ttm_pytorch python=3.11 -y
conda activate ttm_pytorch

# Install PyTorch using the command appropriate for your machine from pytorch.org.
# For a simple CPU environment:
pip install torch

cd ttm_pytorch
python test_ttm.py
```

If you already installed a CUDA-enabled PyTorch build, no extra dependency is
needed for this code.

## Main tensor shapes

One step:

```text
memory M_t:   [B, M, C]
input I_t:    [B, N, C]
read Z_t:     [B, R, C]
output O_t:   [B, R, C]
memory M_t+1: [B, M, C]
```

Sequence:

```text
input:  [B, T, N, C]
output: [B, T, R, C]
```

## Minimal use

```python
import torch
from token_turing import TokenTuringMachineEncoder

B, T, N, C = 2, 6, 32, 64

x = torch.randn(B, T, N, C)

ttm = TokenTuringMachineEncoder(
    dim=C,
    process_size=8,
    memory_size=64,
    memory_mode="TL",
    processing_unit="transformer",
    num_layers=4,
    mlp_dim=512,
    num_heads=8,
)

y, memory = ttm(x, return_memory=True)

print(y.shape)       # [2, 6, 8, 64]
print(memory.shape)  # [2, 64, 64]
```

## Streaming use

```python
memory = ttm.init_memory(batch_size=B)

for t in range(T):
    memory, output_t = ttm.forward_step(x[:, t], memory)
```

The same memory tensor is carried from one step to the next. Reset it only when
you want to start an independent sequence/video.

## Notes

The default and most faithful path is:

```text
memory_mode="TL"
processing_unit="transformer"
```

This implements the paper's recurrent flow:

```text
Z_t     = Read(M_t, I_t)
O_t     = Process(Z_t)
M_t+1   = Write(M_t, O_t, I_t)
```

Both Read and Write use learned token summarization.

`TL-MHA` is also included. `TL-AddErase` is included as an experimental
NTM-style write alternative.

The standalone port intentionally does not depend on JAX, Flax, Scenic, ViViT,
or ml_collections.
