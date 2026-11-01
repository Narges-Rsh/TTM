"""PyTorch implementation of Token Turing Machines (TTM).

Ported from Google Research Scenic:
  scenic/projects/token_turing/model.py
and the TokenLearner V1.1 module used by TTM:
  scenic/projects/token_learner/model.py

Expected tensor convention:
    [batch, tokens, channels]
Full encoder input:
    [batch, steps, tokens_per_step, channels]

This module implements:
- TokenLearnerV11 (MLP token summarizer)
- TokenLearnerMHA (latent-query/MHA summarizer)
- TokenAddEraseWrite
- Transformer / MLP processing units
- TokenTuringMachineUnit
- TokenTuringMachineSimpleUnit
- TokenTuringMachineEncoder
- stateful streaming inference via init_memory() + forward_step()
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Scenic-style 2-layer MLP block operating on the last dimension."""

    def __init__(
        self,
        in_dim: int,
        mlp_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TokenLearnerV11(nn.Module):
    """TokenLearner V1.1 used by the official TTM implementation.

    Scenic/JAX logic:
        LN -> MLP(C -> bottleneck -> K)
        -> transpose -> softmax over original token dimension
        -> weighted sum of the input tokens

    Input:
        x: [B, N, C]
    Output:
        [B, K, C]
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.token_masking = MLPBlock(
            in_dim=dim,
            mlp_dim=bottleneck_dim,
            out_dim=num_tokens,
            dropout=dropout,
        )

    def forward(self, x: Tensor, return_attention: bool = False):
        if x.ndim != 3:
            raise ValueError(f"TokenLearnerV11 expects [B,N,C], got {tuple(x.shape)}")

        selected = self.token_masking(self.norm(x))      # [B, N, K]
        selected = selected.transpose(1, 2)              # [B, K, N]
        attention = F.softmax(selected, dim=-1)          # summarize across N
        tokens = torch.einsum("bkn,bnc->bkc", attention, x)

        if return_attention:
            return tokens, attention
        return tokens


class TokenLearnerMHA(nn.Module):
    """Latent-query/MHA token summarizer from the official TTM code.

    Learned K query tokens attend to all N input tokens.

    Input:  [B, N, C]
    Output: [B, K, C]
    """

    def __init__(self, dim: int, num_tokens: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_tokens = num_tokens
        self.tokens = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.normal_(self.tokens, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: Tensor, return_attention: bool = False):
        if x.ndim != 3:
            raise ValueError(f"TokenLearnerMHA expects [B,N,C], got {tuple(x.shape)}")
        b = x.shape[0]
        queries = self.tokens.expand(b, -1, -1)
        out, attn = self.attn(
            query=queries,
            key=x,
            value=x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        if return_attention:
            return out, attn
        return out


class LearnedPositionEmbedding(nn.Module):
    """Learned positional embedding for a fixed maximum token count.

    The official Flax implementation creates embeddings from the input shape.
    In PyTorch we specify a maximum length up front so parameters exist before
    the optimizer is constructed.
    """

    def __init__(self, max_tokens: int, dim: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.pos = nn.Parameter(torch.empty(1, max_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[1]
        if n > self.max_tokens:
            raise ValueError(
                f"Sequence has {n} tokens but positional embedding supports "
                f"only {self.max_tokens}."
            )
        return x + self.pos[:, :n]


class TransformerProcessingUnit(nn.Module):
    """ViT-like pre-norm Transformer encoder."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class ResidualMLPProcessingUnit(nn.Module):
    """Simple token-wise MLP alternative.

    The paper allows generic processing units; Transformer is the default.
    This implementation keeps a residual path for stable PyTorch training.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_layers))
        self.mlps = nn.ModuleList(
            MLPBlock(dim, mlp_dim, dim, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for norm, mlp in zip(self.norms, self.mlps):
            x = x + mlp(norm(x))
        return self.final_norm(x)


class TokenAddEraseWrite(nn.Module):
    """NTM-inspired differentiable add/erase write.

    This follows the structure of Scenic's TokenAddEraseWrite while making
    dimensions explicit for PyTorch.

    memory:         [B, M, C]
    control_inputs: [B, R, C]
    output:         [B, M, C]
    """

    def __init__(
        self,
        dim: int,
        process_size: int,
        num_tokens: int = 8,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens

        # Select memory locations: C -> bottleneck -> K
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_selector = MLPBlock(
            dim, bottleneck_dim, num_tokens, dropout
        )

        # Scenic transposes control [B,R,C] -> [B,C,R], therefore its MLP
        # operates over R. We reproduce that explicitly.
        self.erase_control_norm = nn.LayerNorm(dim)
        self.erase_token_projection = MLPBlock(
            process_size, bottleneck_dim, num_tokens, dropout
        )
        self.erase_feature_mlp = MLPBlock(
            dim, bottleneck_dim, dim, dropout
        )

        self.add_control_norm = nn.LayerNorm(dim)
        self.add_token_projection = MLPBlock(
            process_size, bottleneck_dim, num_tokens, dropout
        )
        self.add_feature_mlp = MLPBlock(
            dim, bottleneck_dim, dim, dropout
        )

    def _control_vectors(
        self,
        control: Tensor,
        norm: nn.LayerNorm,
        token_proj: nn.Module,
        feature_mlp: nn.Module,
    ) -> Tensor:
        # [B,R,C] -> normalize C -> [B,C,R]
        x = norm(control).transpose(1, 2)
        # [B,C,R] -> [B,C,K] -> [B,K,C]
        x = token_proj(x).transpose(1, 2)
        # feature-wise transform [B,K,C]
        x = feature_mlp(x)
        return x

    def forward(self, memory: Tensor, control_inputs: Tensor) -> Tensor:
        # Soft selection over M memory slots for each of K write heads.
        selected = self.memory_selector(self.memory_norm(memory))  # [B,M,K]
        selected = F.softmax(selected.transpose(1, 2), dim=-1)     # [B,K,M]

        erase = self._control_vectors(
            control_inputs,
            self.erase_control_norm,
            self.erase_token_projection,
            self.erase_feature_mlp,
        )                                                         # [B,K,C]
        erase = torch.sigmoid(erase)

        # Product across K heads as in NTM-style erase.
        wet = selected.unsqueeze(-1) * erase.unsqueeze(2)          # [B,K,M,C]
        erase_gate = torch.prod(1.0 - wet, dim=1)                  # [B,M,C]
        output = memory * erase_gate

        add = self._control_vectors(
            control_inputs,
            self.add_control_norm,
            self.add_token_projection,
            self.add_feature_mlp,
        )                                                         # [B,K,C]

        # Add contributions weighted by selected memory slots.
        add_term = (
            selected.unsqueeze(-1) * add.unsqueeze(2)
        ).mean(dim=1)                                              # [B,M,C]
        output = output + add_term
        return output


class TokenTuringMachineUnit(nn.Module):
    """One recurrent TTM step.

    Read:
        Z_t = S_R([M_t || I_t])

    Process:
        O_t = Process(Z_t)

    Write:
        M_{t+1} = S_M([M_t || I_t || O_t])

    Input:
        memory_tokens: [B, M, C]
        input_tokens:  [B, N, C]

    Returns:
        new_memory:    [B, M, C]
        output_tokens: [B, R, C]
    """

    def __init__(
        self,
        dim: int,
        process_size: int = 8,
        memory_size: int = 64,
        memory_mode: str = "TL",
        processing_unit: str = "transformer",
        num_layers: int = 1,
        mlp_dim: int = 512,
        num_heads: int = 8,
        use_positional_embedding: bool = False,
        max_input_tokens: Optional[int] = None,
        dropout: float = 0.0,
        tokenlearner_bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.process_size = process_size
        self.memory_size = memory_size
        self.memory_mode = memory_mode
        self.processing_unit_name = processing_unit
        self.use_positional_embedding = use_positional_embedding

        valid_memory_modes = {"TL", "TL-MHA", "TL-AddErase"}
        if memory_mode not in valid_memory_modes:
            raise ValueError(
                f"memory_mode must be one of {valid_memory_modes}, got {memory_mode}"
            )

        if memory_mode in {"TL", "TL-AddErase"}:
            self.read_summarizer = TokenLearnerV11(
                dim=dim,
                num_tokens=process_size,
                bottleneck_dim=tokenlearner_bottleneck_dim,
                dropout=dropout,
            )
        else:
            self.read_summarizer = TokenLearnerMHA(
                dim=dim, num_tokens=process_size, num_heads=num_heads
            )

        if processing_unit == "transformer":
            self.processor = TransformerProcessingUnit(
                dim=dim,
                num_layers=num_layers,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif processing_unit == "mlp":
            self.processor = ResidualMLPProcessingUnit(
                dim=dim,
                num_layers=num_layers,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                "This standalone PyTorch port currently supports "
                "processing_unit='transformer' or 'mlp'."
            )

        if memory_mode == "TL":
            self.write_summarizer = TokenLearnerV11(
                dim=dim,
                num_tokens=memory_size,
                bottleneck_dim=tokenlearner_bottleneck_dim,
                dropout=dropout,
            )
        elif memory_mode == "TL-MHA":
            self.write_summarizer = TokenLearnerMHA(
                dim=dim, num_tokens=memory_size, num_heads=num_heads
            )
        else:
            self.write_summarizer = TokenAddEraseWrite(
                dim=dim,
                process_size=process_size,
                num_tokens=8,
                bottleneck_dim=tokenlearner_bottleneck_dim,
                dropout=dropout,
            )

        if use_positional_embedding:
            if max_input_tokens is None:
                raise ValueError(
                    "Set max_input_tokens when use_positional_embedding=True."
                )
            self.read_pos_embed = LearnedPositionEmbedding(
                memory_size + max_input_tokens, dim
            )
            self.write_pos_embed = LearnedPositionEmbedding(
                memory_size + max_input_tokens + process_size, dim
            )
        else:
            self.read_pos_embed = nn.Identity()
            self.write_pos_embed = nn.Identity()

    def forward(
        self,
        memory_tokens: Tensor,
        input_tokens: Tensor,
        return_debug: bool = False,
    ):
        if memory_tokens.ndim != 3 or input_tokens.ndim != 3:
            raise ValueError("memory_tokens and input_tokens must both be [B,N,C].")
        if memory_tokens.shape[0] != input_tokens.shape[0]:
            raise ValueError("Batch sizes of memory and input must match.")
        if memory_tokens.shape[-1] != self.dim or input_tokens.shape[-1] != self.dim:
            raise ValueError(f"Expected channel dim C={self.dim}.")
        if memory_tokens.shape[1] != self.memory_size:
            raise ValueError(
                f"Expected {self.memory_size} memory tokens, "
                f"got {memory_tokens.shape[1]}."
            )

        # -------- READ --------
        read_source = torch.cat([memory_tokens, input_tokens], dim=1)
        read_source = self.read_pos_embed(read_source)

        if return_debug and isinstance(self.read_summarizer, TokenLearnerV11):
            read_tokens, read_attention = self.read_summarizer(
                read_source, return_attention=True
            )
        elif return_debug and isinstance(self.read_summarizer, TokenLearnerMHA):
            read_tokens, read_attention = self.read_summarizer(
                read_source, return_attention=True
            )
        else:
            read_tokens = self.read_summarizer(read_source)
            read_attention = None

        # -------- PROCESS --------
        output_tokens = self.processor(read_tokens)

        # -------- WRITE --------
        if self.memory_mode == "TL-AddErase":
            new_memory = self.write_summarizer(memory_tokens, output_tokens)
            write_attention = None
        else:
            write_source = torch.cat(
                [memory_tokens, input_tokens, output_tokens], dim=1
            )
            write_source = self.write_pos_embed(write_source)

            if return_debug and isinstance(self.write_summarizer, TokenLearnerV11):
                new_memory, write_attention = self.write_summarizer(
                    write_source, return_attention=True
                )
            elif return_debug and isinstance(self.write_summarizer, TokenLearnerMHA):
                new_memory, write_attention = self.write_summarizer(
                    write_source, return_attention=True
                )
            else:
                new_memory = self.write_summarizer(write_source)
                write_attention = None

        if return_debug:
            debug = {
                "read_source": read_source,
                "read_tokens": read_tokens,
                "read_attention": read_attention,
                "output_tokens": output_tokens,
                "new_memory": new_memory,
                "write_attention": write_attention,
            }
            return new_memory, output_tokens, debug

        return new_memory, output_tokens


class TokenTuringMachineSimpleUnit(nn.Module):
    """Simplified TTM unit.

    TokenLearner first reduces current input to R tokens. Those R tokens are
    concatenated with M memory tokens and jointly processed.

    This port uses the intuitive ordering [input_summary || memory], so the
    first R outputs are the current output tokens and the remaining M outputs
    are the new memory.
    """

    def __init__(
        self,
        dim: int,
        process_size: int = 16,
        memory_size: int = 96,
        processing_unit: str = "transformer",
        num_layers: int = 1,
        mlp_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.process_size = process_size
        self.memory_size = memory_size

        self.input_summarizer = TokenLearnerV11(
            dim=dim,
            num_tokens=process_size,
            bottleneck_dim=64,
            dropout=dropout,
        )

        if processing_unit == "transformer":
            self.processor = TransformerProcessingUnit(
                dim, num_layers, mlp_dim, num_heads, dropout
            )
        elif processing_unit == "mlp":
            self.processor = ResidualMLPProcessingUnit(
                dim, num_layers, mlp_dim, dropout
            )
        else:
            raise ValueError("SimpleUnit supports 'transformer' or 'mlp' here.")

    def forward(self, memory_tokens: Tensor, input_tokens: Tensor):
        current = self.input_summarizer(input_tokens)             # [B,R,C]
        all_tokens = torch.cat([current, memory_tokens], dim=1)   # [B,R+M,C]
        processed = self.processor(all_tokens)
        output_tokens = processed[:, : self.process_size]
        new_memory = processed[:, self.process_size :]
        return new_memory, output_tokens


class TokenTuringMachineEncoder(nn.Module):
    """TTM over a fixed sequence plus a streaming one-step API.

    Full forward input:
        [B, T, N, C]

    Full forward output:
        [B, T, R, C]

    Memory is initialized to zeros at the beginning of each independent
    sequence, exactly matching the official fixed-sequence encoder design.
    """

    def __init__(
        self,
        dim: int,
        process_size: int = 8,
        memory_size: int = 64,
        memory_mode: str = "TL",
        processing_unit: str = "transformer",
        num_layers: int = 4,
        mlp_dim: int = 512,
        num_heads: int = 8,
        use_positional_embedding: bool = False,
        max_input_tokens: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.process_size = process_size
        self.memory_size = memory_size

        self.unit = TokenTuringMachineUnit(
            dim=dim,
            process_size=process_size,
            memory_size=memory_size,
            memory_mode=memory_mode,
            processing_unit=processing_unit,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            use_positional_embedding=use_positional_embedding,
            max_input_tokens=max_input_tokens,
            dropout=dropout,
        )

    def init_memory(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        p = next(self.parameters())
        device = p.device if device is None else device
        dtype = p.dtype if dtype is None else dtype
        return torch.zeros(
            batch_size, self.memory_size, self.dim,
            device=device, dtype=dtype
        )

    def forward_step(
        self,
        input_tokens: Tensor,
        memory_tokens: Optional[Tensor] = None,
        return_debug: bool = False,
    ):
        """Process exactly one time step for streaming inference."""
        if input_tokens.ndim != 3:
            raise ValueError(
                f"forward_step expects [B,N,C], got {tuple(input_tokens.shape)}"
            )
        if memory_tokens is None:
            memory_tokens = self.init_memory(
                input_tokens.shape[0],
                device=input_tokens.device,
                dtype=input_tokens.dtype,
            )
        return self.unit(
            memory_tokens,
            input_tokens,
            return_debug=return_debug,
        )

    def forward(
        self,
        input_tokens: Tensor,
        initial_memory: Optional[Tensor] = None,
        return_memory: bool = False,
    ):
        if input_tokens.ndim != 4:
            raise ValueError(
                f"Expected [B,T,N,C], got {tuple(input_tokens.shape)}"
            )

        b, t, _, c = input_tokens.shape
        if c != self.dim:
            raise ValueError(f"Expected C={self.dim}, got C={c}")

        if initial_memory is None:
            memory = self.init_memory(
                b, device=input_tokens.device, dtype=input_tokens.dtype
            )
        else:
            memory = initial_memory

        outputs = []
        for step in range(t):
            memory, out = self.unit(memory, input_tokens[:, step])
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [B,T,R,C]

        if return_memory:
            return outputs, memory
        return outputs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
