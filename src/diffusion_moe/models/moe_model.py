"""Diffusion-routed MoE transformer: BaseTransformer with a configurable
subset of layers replaced by a MoE layer, for ablating how many/which layers
benefit from diffusion routing, and which router type (Section 11 baselines)."""

from __future__ import annotations

import math

import torch
from torch import nn

from diffusion_moe.models.cosine_moe_layer import CosineMoELayer
from diffusion_moe.models.moe_layer import DiffusionMoELayer
from diffusion_moe.models.random_moe_layer import RandomMoELayer
from diffusion_moe.models.rmsnorm import RMSNorm
from diffusion_moe.models.switch_moe_layer import SwitchMoELayer
from diffusion_moe.models.transformer_block import TransformerBlock

_ROUTER_CLASSES = {
    "diffusion": DiffusionMoELayer,
    "cosine": CosineMoELayer,
    "switch": SwitchMoELayer,
    "random": RandomMoELayer,
}


def _build_moe_layer(
    router: str,
    d_model: int,
    n_heads: int,
    max_seq_len: int,
    n_experts: int,
    top_k: int,
    n_components: int,
    n_landmarks: int,
    diffusion_t: int,
    alpha: float,
    tau: float,
    ffn_dim: int | None,
    overlap_factor: float,
    rope_base: int,
    norm_eps: float,
    dropout: float,
    centroid_refresh_steps: int,
    noise_std: float = 0.0,
    cosine: bool = False,
) -> nn.Module:
    """Builds one MoE layer of the requested router type. Each variant takes
    only the subset of these hyperparameters that are meaningful for it (e.g.
    only "diffusion" uses n_components/n_landmarks/diffusion_t/alpha/
    centroid_refresh_steps/noise_std/cosine; only "diffusion" and "cosine"
    use tau). `cosine` here means DiffusionMoELayer's own kernel-metric
    choice for its diffusion map (raw-Euclidean vs. L2-normalized/cosine
    activations before fitting) -- unrelated to the separate `router="cosine"`
    variant (CosineMoELayer), which routes by cosine similarity directly in
    raw d_model space with no diffusion map at all. Same word, two different
    things at different stages of the pipeline -- see architecture.md §5."""
    if router not in _ROUTER_CLASSES:
        raise ValueError(f"Unknown router '{router}'. Supported: {sorted(_ROUTER_CLASSES)}")

    common = dict(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=max_seq_len,
        n_experts=n_experts,
        top_k=top_k,
        ffn_dim=ffn_dim,
        overlap_factor=overlap_factor,
        rope_base=rope_base,
        norm_eps=norm_eps,
        dropout=dropout,
    )

    if router == "diffusion":
        return DiffusionMoELayer(
            **common,
            n_components=n_components,
            n_landmarks=n_landmarks,
            diffusion_t=diffusion_t,
            alpha=alpha,
            tau=tau,
            centroid_refresh_steps=centroid_refresh_steps,
            noise_std=noise_std,
            cosine=cosine,
        )
    if router == "cosine":
        return CosineMoELayer(**common, tau=tau)
    if router == "switch":
        return SwitchMoELayer(**common)
    return RandomMoELayer(**common)  # router == "random"


class DiffusionMoETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        n_experts: int,
        top_k: int,
        n_components: int,
        n_landmarks: int,
        diffusion_t: int = 3,
        alpha: float = 1.0,
        tau: float = 0.1,
        ffn_dim: int | None = None,
        overlap_factor: float = 1.0,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        centroid_refresh_steps: int = 500,
        router: str = "diffusion",
        layers_to_replace: list[int] | None = None,
        noise_std: float = 0.0,
        cosine_layers: list[int] | None = None,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Default: replace every layer (a "full" diffusion-MoE model).
        replace = set(range(n_layers)) if layers_to_replace is None else set(layers_to_replace)
        for idx in replace:
            if not 0 <= idx < n_layers:
                raise ValueError(f"layers_to_replace index {idx} out of range [0, {n_layers})")
        self.layers_to_replace = replace

        # Per-layer, not a project-wide switch -- see the recommendation
        # this implements in _build_moe_layer's docstring. Layer indices
        # here that aren't also in layers_to_replace are simply inert (no
        # DiffusionMoELayer gets built there to apply it to).
        cosine_set = set(cosine_layers) if cosine_layers else set()
        for idx in cosine_set:
            if not 0 <= idx < n_layers:
                raise ValueError(f"cosine_layers index {idx} out of range [0, {n_layers})")
        self.cosine_layers = cosine_set

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # nn.Embedding's default init is Normal(0, 1) -- std=1.0, dramatically
        # too large for a transformer embedding table, and especially costly
        # here since tie_embeddings=True by default makes this same
        # oversized matrix double as the LM-head unembedding projection.
        # This turned out to be the dominant contributor to the ~828 max
        # abs logit / ~823 nats initial task_loss anomaly this whole
        # investigation started from -- not residual-stream depth-compounding
        # (see the scaled out_proj/down_proj init below, which fixed that
        # part but left logits *unchanged* until this was also fixed).
        # 0.02 is the standard GPT-2/nanoGPT convention.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        blocks: list[nn.Module] = []
        for layer_idx in range(n_layers):
            if layer_idx in replace:
                blocks.append(
                    _build_moe_layer(
                        router,
                        d_model,
                        n_heads,
                        max_seq_len,
                        n_experts=n_experts,
                        top_k=top_k,
                        n_components=n_components,
                        n_landmarks=n_landmarks,
                        diffusion_t=diffusion_t,
                        alpha=alpha,
                        tau=tau,
                        ffn_dim=ffn_dim,
                        overlap_factor=overlap_factor,
                        rope_base=rope_base,
                        norm_eps=norm_eps,
                        dropout=dropout,
                        centroid_refresh_steps=centroid_refresh_steps,
                        noise_std=noise_std,
                        cosine=layer_idx in cosine_set,
                    )
                )
            else:
                blocks.append(
                    TransformerBlock(
                        d_model,
                        n_heads,
                        max_seq_len,
                        ffn_dim=ffn_dim,
                        rope_base=rope_base,
                        norm_eps=norm_eps,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Depth-aware residual-branch output scaling (GPT-2/nanoGPT
        # convention, absent here until now): every residual sub-block's
        # OUTPUT projection -- attention's out_proj, every FFN's down_proj
        # (dense TransformerBlocks, every routed ExpertFFN, and the
        # mandatory shared_expert alike) -- gets its default-initialized
        # weight scaled down by 1/sqrt(2*n_layers). Without this, residual-
        # stream variance compounds with depth; confirmed empirically on
        # this exact 24-layer/every-layer-MoE config before this fix: random
        # init produced logits with max abs ~828, matching almost exactly
        # the ~823 nats initial task_loss it was causing (ln(vocab_size)=
        # 10.4 is what a properly-calibrated random init should give). The
        # "2" accounts for each block contributing two residual additions
        # (attention, then FFN/MoE). Matched on the Linear's own attribute
        # name (out_proj/down_proj) rather than its module class, since
        # ExpertFFN is shared by routed and shared experts alike and
        # TransformerBlock's FeedForward is a separate class entirely --
        # this scopes to exactly the residual-branch outputs regardless of
        # which of those it's nested in.
        residual_output_scale = 1.0 / math.sqrt(2 * n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in (
                "out_proj",
                "down_proj",
            ):
                with torch.no_grad():
                    module.weight.mul_(residual_output_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
        """Returns (logits, activations, router_outputs).

        activations[i] is the residual-stream output of layer i, for every
        layer (dense or MoE) — same convention as BaseTransformer. router_outputs
        holds the routing aux dict (router_logits, gate_values, expert_indices,
        Psi_t) for each MoE layer only, keyed by layer index, for the
        load-balance/separation losses in Section 8.
        """
        batch, seq_len = input_ids.shape
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch, -1)

        x = self.token_embedding(input_ids)

        activations: dict[int, torch.Tensor] = {}
        router_outputs: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx in self.layers_to_replace:
                x, aux = block(x, positions, mask)
                router_outputs[layer_idx] = aux
            else:
                x = block(x, positions, mask)
            activations[layer_idx] = x

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, activations, router_outputs
