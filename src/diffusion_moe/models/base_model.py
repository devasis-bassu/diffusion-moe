"""Dense baseline transformer: token embedding -> N TransformerBlocks -> RMSNorm -> LM head."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_moe.models.rmsnorm import RMSNorm
from diffusion_moe.models.transformer_block import TransformerBlock


class BaseTransformer(nn.Module):
    """Dense (non-MoE) transformer. Serves as the compute baseline in Section 11
    and the scaffold that DiffusionMoETransformer (Section 7) replaces layers of."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        ffn_dim: int | None = None,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    max_seq_len,
                    ffn_dim=ffn_dim,
                    rope_base=rope_base,
                    norm_eps=norm_eps,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Returns (logits, activations) where activations[i] is the residual-stream
        output of layer i — the per-layer hidden states consumed by geometry
        extraction (Section 9) and by DiffusionMoELayer's routing (Section 7)."""
        batch, seq_len = input_ids.shape
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch, -1)

        x = self.token_embedding(input_ids)

        activations: dict[int, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, positions, mask)
            activations[layer_idx] = x

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, activations
