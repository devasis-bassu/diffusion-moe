"""Builds a DiffusionMoETransformer from a Hydra config — shared by
scripts/train.py and scripts/evaluate.py so model construction stays in sync."""

from __future__ import annotations

from typing import Any

from diffusion_moe.models.moe_model import DiffusionMoETransformer


def build_model_from_config(cfg: Any) -> DiffusionMoETransformer:
    return DiffusionMoETransformer(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        max_seq_len=cfg.model.max_seq_len,
        n_experts=cfg.routing.n_experts,
        top_k=cfg.routing.top_k,
        n_components=cfg.routing.n_components,
        n_landmarks=cfg.routing.n_landmarks,
        diffusion_t=cfg.routing.diffusion_t,
        alpha=cfg.routing.alpha,
        tau=cfg.routing.tau,
        ffn_dim=cfg.model.ffn_dim,
        rope_base=cfg.model.rope_base,
        norm_eps=cfg.model.norm_eps,
        centroid_refresh_steps=cfg.routing.centroid_refresh_steps,
        router=cfg.model.get("router", "diffusion"),
        layers_to_replace=cfg.get("layers_to_replace", None),
        noise_std=cfg.routing.get("noise_std", 0.0),
        cosine_layers=cfg.get("cosine_layers", None),
        grad_accum_steps=cfg.training.grad_accum_steps,
    )
