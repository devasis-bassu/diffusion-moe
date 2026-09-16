# Diffusion-MoE Phase 2 Training Investigation

### Findings report — all-layers training run, bugs found, and current state

*Prepared for review · diffusion-moe project*

---

## Executive Summary

Following Phase 1's diagnostics (`phase1_findings_report.md`) and the bounded single-layer pilot fine-tune, this phase asked the question those tools structurally couldn't answer: **what actually happens if every layer of the project's own from-scratch architecture is a `DiffusionMoELayer`, trained end-to-end, not just diagnosed or spliced into a frozen pretrained backbone?**

The honest answer is: five real, previously-undiscovered bugs, one badly-miscalibrated hyperparameter, and — after fixing all of them — one clean, complete training run that demonstrates the architecture can learn, route without permanently collapsing, and finish without crashing. That run is real progress, not a finished result: see §7 for why it is not yet evidence for scaling up.

1. **NaN divergence at step ~199** — `NystromDiffusionMap`'s kernel bandwidth (`eps_`) is fit once and frozen between landmark refits; once training moves the embedding distribution far enough, a token can drift outside every landmark's kernel support, and the unguarded division silently produces NaN, which `clip_grad_norm_` cannot repair. Fixed with both a defensive guard and — the deeper fix — a live `eps_` refresh between refits (§2.1).
2. **Checkpoint resume crashes a real MoE layer** — `NystromDiffusionMap`'s fitted state lives on a plain Python object, not an `nn.Module`, so it silently does not survive `state_dict()` save/load. Any resume off the exact refit-schedule boundary crashed outright. Found running the project's own `expert_attribution.py` tool against a real checkpoint, not by inspection (§2.2).
3. **No shared/always-active expert** — every FFN computation was contingent on sparse top-k routing; nothing had guaranteed, routing-independent gradient signal. Added a mandatory `shared_expert`, architecturally identical to a routed `ExpertFFN`, applied unconditionally outside the router (§3).
4. **Disk-full crash near the finish line** — unbounded checkpoint retention filled a 60GB instance's disk with ~5.7GB files, corrupting the final checkpoint write. Fixed with automatic pruning (§4).
5. **A catastrophic initialization bug** — `task_loss` started at ~823 nats against a random-guessing baseline of `ln(32000)=10.37`, and never recovered below ~60 nats across any run, no matter how long training continued. Root-caused to two compounding, previously-unaddressed initialization gaps (no custom init existed anywhere in the model) and fixed; confirmed both in isolation and in a real training run (§5). This is the most consequential finding in this phase — everything observed in earlier runs (§6) was training on top of this bug.
6. **`mu_load` (the load-balance loss weight) was miscalibrated by ~2 orders of magnitude** relative to `task_loss`'s gradient once the loss scale was actually correct — measured directly, not guessed, and retuned (§6.4).

The final run (`graceful-universe-21`) completed its full 2,441-step / 20M-token budget cleanly: zero NaN, `task_loss` ending at 7.08 (below the random baseline), all 24 layers' `load_loss` ending near zero (down from 7 of 24 near-total collapse mid-run), and `val_ppl` finally moving off its `exp(20)` safety cap for the first time in this investigation (7,286 → 5,081 → 35,294 spike → 16,158 → 3,245 → 2,239 → 974). See §7 for why this is "the infrastructure works" evidence, not yet "the architecture is validated" evidence.

---

## 1. Background and Motivation

Phase 1's diagnostics and the bounded pilot fine-tune (`phase1_findings_report.md` §3.4–3.6) each tested one layer at a time, spliced into an otherwise-frozen pretrained Mistral-7B backbone. That structurally cannot answer a sharper question raised during review: *if a layer's router keeps forcing collapse onto one expert during real training, is that proof the layer can't benefit from diffusion routing at all, or just proof that one frozen-backbone, one-layer splice hasn't given the router anything to differentiate?* The only way to find out is to train the project's own from-scratch architecture with every layer routed, and watch. The user's explicit direction: *"I would let all layers use diffusion MoE and see what happens."*

This phase covers that run, the infrastructure it required (per-layer loss breakdown, checkpoint resume, expert-attribution analytics — see `methodology.md` §5 and §6 for the durable reference material), the bugs it surfaced, and the final result.

All runs in this section use the project's own 300M/24-layer from-scratch architecture (`configs/model/300m.yaml`), trained on `wikitext-103-raw-v1` via local parquet files (bypassing the recurring flaky-CDN Hub-streaming stall — see `methodology.md` §4), on a single rented RTX 3090 instance (vast.ai).

---

## 2. Bug: NaN Divergence, and the Deeper Fix Behind It

### 2.1 Root cause: a frozen kernel bandwidth under a moving distribution

The first all-layers run diverged to NaN at step 199, confirmed via an explicit `RuntimeWarning: invalid value encountered in divide` in `geometry/nystrom.py`. `NystromDiffusionMap.transform()`'s Nyström extension formula divides by each new point's total kernel mass to the (frozen) landmarks:

```python
K_alpha = K_new / (d_alpha_new[:, None] * self._d_alpha_landmarks[None, :])
```

`eps_` (the kernel bandwidth) is derived once from the landmarks' own spread at the last refit (`fit()`, only run every `centroid_refresh_steps`) and then frozen until the next one. If a token's embedding drifts far enough from that frozen reference — plausible once `centroid_separation_loss` has been actively pushing centroids apart for hundreds of steps — its kernel value to *every* landmark can underflow to exactly `0.0` in float64 (`exp(v)` underflows to the literal bit pattern `0.0` below roughly `v < -745`, not a small-but-nonzero value). The resulting `0/0` division produced `NaN`, which propagated through `task_loss`/`load_loss`, and — critically — `torch.nn.utils.clip_grad_norm_` cannot repair a NaN gradient (the norm of a NaN tensor is itself NaN), so every parameter update after that point silently corrupted the entire model. Training continued to "run" for ~40 more steps on fully poisoned weights before crashing outright on an unrelated downstream `KMeans` call (sklearn's input validator rejects NaN explicitly; the raw numpy division earlier did not).

**Immediate fix**: guard the division (`d_alpha_new_safe`, mirroring the existing `row_sums` guard a few lines below) so a token with zero kernel mass gets diffusion coordinates at the origin (maximal uncertainty) instead of NaN.

### 2.2 The real fix: live-refreshing the kernel bandwidth between refits

The guard alone treats a symptom. The actual staleness problem — `eps_` describing a step-0 snapshot of the embedding distribution, growing more wrong the longer training runs between refits — remains. Fixed by having `transform()` re-derive `eps_` from the *current* batch's actual distance to the (still-frozen) landmarks on every call, reusing the distance matrix already computed for the Nyström kernel (cheap — no new k-means).

This surfaced two further correctness requirements, both needed for the fix to be mathematically sound rather than merely avoid a crash:

- **`eigenvectors_`/`psi_landmarks_` must be re-derived alongside `eps_`** (extracted into a shared `_fit_landmark_spectrum`, called by both `fit()` and the refresh path): Nyström's defining correctness property — that applying the extension formula to the landmarks themselves must exactly reproduce their direct diffusion coordinates — only holds when the landmark spectrum and the new-point extension share one consistent kernel. Caught by `test_nystrom_quality_exact_reproduction_on_landmarks` when a first attempt (refreshing only `eps_` in isolation) broke it (0.0042 error against a 1e-4 tolerance).
- **`diffusion_eigenvectors`'s ARPACK solve needed a seeded starting vector.** It previously ran once per landmark set, so its default unseeded random start never mattered. Once it could rerun every step, an unseeded start would make the diffusion-coordinate basis jitter step-to-step from solver randomness alone — independent of any real drift in the data, i.e. the opposite of what this fix was for.
- **`fit_transform`'s own internal `transform()` call must skip the refresh** (new `_refresh_eps` parameter): it runs immediately after `fit()` on the same data, so there is no staleness yet to correct, and refreshing anyway would silently override `fit()`'s landmark-based heuristic and redo the eigensolve twice on every refit step for no benefit.

**Cost**: this is not free, but it is cheap relative to what it replaces. `eps_`'s refresh reuses an already-computed distance matrix; the re-derived eigensolve is over just the `n_landmarks × n_landmarks` matrix (32–128 points), not another k-means clustering over the full token batch. Measured directly in production: steady-state per-step time rose from ~5.55s to ~7.0–7.3s (roughly 25-30%), on top of an already CPU-bound geometry pipeline (this entire subsystem runs on `z.detach()`'d numpy/sklearn, never touching the GPU).

---

## 3. Feature: A Mandatory Shared Expert, Outside the Router

Once every layer was routed and training could be observed cleanly, a striking pattern emerged (visible even before the initialization bug was found, §5): `load_loss` stayed elevated across nearly all 24 layers simultaneously — not one collapsed outlier, a near-universal, if partial, imbalance (mean `load_loss` 3.21 across all layers at one snapshot, only 1 of 24 layers below 1.0). This raised the mechanistic question motivating this whole run: with *every* FFN computation contingent on sparse top-k routing, nothing in the network had guaranteed, routing-independent gradient signal — common/generic computation had to be either redundantly relearned by whichever expert won that step's routing lottery, or not learned reliably at all if routing was noisy.

This is a known pattern in production MoE architectures (DeepSeekMoE's "shared expert isolation," similarly Qwen2-MoE), added here as `DiffusionMoELayer.shared_expert`: architecturally identical to a routed `ExpertFFN` (same width, same `overlap_factor`), applied unconditionally to every token, every step — no gate value, no centroid, entirely outside the diffusion router's influence. This adds real active compute per token (`top_k + 1` experts fire instead of `top_k`), not a reallocation of existing capacity.

Verified with a direct behavioral test (`test_shared_expert_contributes_even_when_routing_contributes_nothing`): with the routed dispatch path monkeypatched to contribute exactly zero, the layer's output still differs from the plain attention residual — the shared expert's contribution cannot be made routing-contingent by construction.

---

## 4. Bug: Disk-Full Crash Near the Finish Line

A run that had otherwise reached step 2,400 of 2,441 (98.3%) crashed on its own checkpoint save: the instance's 60GB disk filled completely (8 accumulated checkpoints × ~5.7GB each), and the final write was cut off mid-file (`PytorchStreamWriter failed writing file`; the resulting `step_2400.pt` was truncated to ~2.97GB against an expected ~5.68GB — corrupt, unusable). The last *valid* checkpoint (`step_2100.pt`, saved before the disk filled) was confirmed to load cleanly.

Fixed with automatic checkpoint retention in `Trainer`: after each numbered checkpoint save, all but the most recent `keep_last_n_checkpoints` (default 2, configurable via `training.keep_last_n_checkpoints`) are deleted. `best.pt` is a separate file and always kept regardless. `<= 0` disables pruning (old, unbounded behavior) for anyone who wants it.

---

## 5. The Initialization Bug: Task_Loss Was Never Actually Calibrated

### 5.1 The anomaly

Every run before this fix showed the same signature: `task_loss` starting around 823 nats (step 1) and never recovering below roughly 60 nats, however long training continued. `val_ppl` (`exp(min(avg_nll, 20.0))`) was pinned at its safety cap — `485,165,195.4` exactly — at *every single logged evaluation, across every run, all the way to the end of a 2,400-step run*. This is a real red flag independent of routing behavior: this project's tokenizer (Mistral's, 32,000 tokens) gives a random-guessing baseline of `ln(32000) ≈ 10.37` nats — the cross-entropy a properly-initialized, completely untrained model should sit near. A trained model that never gets within even a factor of 2 of that baseline, in either direction, is not "undertrained" — something is numerically broken.

### 5.2 Root cause, confirmed by direct measurement

No custom weight initialization existed anywhere in the model — every linear layer and the token embedding relied entirely on PyTorch's defaults. Two compounding gaps, confirmed empirically (not just inferred) by hooking every block of a freshly-constructed 24-layer model and measuring residual-stream norms and final logit magnitude directly:

1. **`token_embedding` used `nn.Embedding`'s default init — `Normal(0, 1)`, std=1.0** — far too large for a transformer embedding table, and especially costly because `tie_embeddings=True` by default makes this same oversized matrix double as the LM-head unembedding projection, directly inflating logit magnitude. This was the *dominant* cause: fixing only the second gap below (residual-output scaling) left final logits unchanged — actually slightly worse (`max abs` 828 → 1071) — until this was also fixed.
2. **No depth-aware scaling on residual-branch output projections** (attention's `out_proj`, every FFN's `down_proj` — dense `TransformerBlock`, routed `ExpertFFN`, and the new `shared_expert` alike). Every serious transformer implementation (GPT-2, LLaMA, nanoGPT) scales these down by `1/sqrt(2 × n_layers)` at init specifically because residual-stream variance otherwise compounds with depth. Confirmed directly: residual-stream norm grew 32 → 44 over 24 layers unfixed; a controlled, smooth 1.0 → 7.1 fixed.

Combined fix: `token_embedding` initialized to the standard `std=0.02` (GPT-2/nanoGPT convention); every `out_proj`/`down_proj` weight scaled by `1/sqrt(2 × n_layers)` after construction (matched on the Linear's own attribute name, so it applies uniformly regardless of which module — dense, routed expert, or shared expert — it's nested in).

**Verified twice.** In isolation: a fresh random-init model's cross-entropy against random labels dropped from ~1,014 to **10.51** (essentially exactly `ln(32000)=10.37`). In real training (`vital-microwave-20`, the first run with this fix): step 1 `task_loss = 10.61`, and a clean, sane trajectory afterward (10.61 → 9.03 → 8.47 by step 26) — nothing resembling any earlier run's trajectory.

### 5.3 What this means for everything observed before this fix

`resilient-brook-18` and `hardy-lake-19` (§6) both trained on top of this bug. Their qualitative conclusions (per-layer collapse patterns, shared-expert comparison) are likely still valid as *relative* signals within those runs, but their absolute loss numbers should not be treated as meaningful, and the shared-expert-vs-not comparison in §6.2 should be re-run now that the loss scale is correct before drawing any firm conclusion from it.

---

## 6. Run-by-Run Summary

| Run (wandb) | Init fix | Shared expert | `mu_load` | Outcome |
|---|---|---|---|---|
| `resilient-brook-18` | No | No | 0.01 | `task_loss` plateaued 60–110 nats; `val_ppl` permanently capped; killed to add the shared expert |
| `hardy-lake-19` | No | Yes | 0.01 | Modest improvement over the above at matched steps (`task_loss=90.3` vs `100.3` @ step 600) but still broken-scale; crashed on disk-full at step 2,400/2,441 (§4) |
| `vital-microwave-20` | **Yes** | Yes | 0.01 | Step 1 `task_loss=10.61` (correct!), but 18 of 24 layers collapsed to `load_loss≈7.0` (the exact CV² ceiling) by step ~600; `task_loss` climbed back to 11.5 — worse than its own starting point; killed for `mu_load` retuning |
| `graceful-universe-21` | **Yes** | Yes | **2.0** | **Completed cleanly** — see §6.3 |

### 6.1 The collapse dynamic got *worse*, not better, once the loss was fixed

`vital-microwave-20` is the sharpest result in this phase. Fixing the initialization did not just fix the loss scale — it let gradients flow cleanly for the first time. `task_loss`'s gradient became a strong, well-calibrated signal instead of one dominated by first fixing a catastrophic starting point, which made the router-collapse shortcut (route everything to one expert per layer, reducing the effective optimization problem) *more* attractive to find, not less. 18 of 24 layers reached essentially the exact `load_loss` ceiling (`n_experts - 1 = 7.0`) within ~600 steps. This is a direct escalation of the question this whole run was designed to answer: collapse is not a rare pathology in one unlucky layer, it is close to the *default* outcome once the network can actually learn efficiently, unless something resists it hard enough.

### 6.2 Measuring, not guessing, the fix

Rather than guess a new `mu_load`, the relative gradient pull of `task_loss` vs. `load_loss` on the shared `ExpertCentroids` parameter was measured directly on a fresh, correctly-initialized model (the parameter both losses actually compete to move, since `load_loss`'s gradient is confined to centroids — `Psi_t` is detached before the routing-loss computation, so it never reaches attention/backbone weights directly). Raw ratio: task_loss's pull was only **1.86×** load_loss's — a fair fight. The old `mu_load=0.01` shrank that to an effective **186×**, leaving `load_loss` almost no real ability to resist collapse — a value very likely tuned against the old, catastrophically-mis-scaled `task_loss` and never revisited once §5's fix corrected it. Set to `mu_load=2.0`, targeting an effective ratio near 1.

### 6.3 The final run

`graceful-universe-21` completed the full 2,441-step budget cleanly — zero NaN, `state=finished`. At the halfway point (step ~296), the picture was a real but partial improvement: 7 of 24 layers near collapse, versus 18 of 24 in the unfixed-`mu` run at a comparable point — down ~60%, not eliminated. By the *end* of training, the picture resolved further: all 24 layers' `load_loss` near zero (best 0.008, most below 0.0005) — though the path there was genuinely volatile, not a clean monotonic decrease (`load_loss` oscillated between 0.45 and 3.6 across steps 1,200–2,200 before settling as the learning rate decayed toward its floor; see §7 for why this specific ending should not yet be over-interpreted).

`task_loss` ended at **7.08** — below the random baseline, genuine learning. `val_ppl` moved for the first time in this entire investigation, off the `exp(20)` cap:

```
step  300:  7,286
step  600:  5,081
step  900: 35,294   <- real spike, not a fluke
step 1200: 16,158
step 1800:  3,245
step 2100:  2,239
step 2400:    975   <- best of the run
```

Non-monotonic and still high in absolute terms for a 300M-parameter model (even on a small budget), but a real, measured signal that the model is generalizing at all — something no earlier run in this entire investigation had shown.

### 6.4 What token-level routing actually looks like (early-training snapshot)

`scripts/expert_attribution.py` (extended this phase with Hydra CLI override support, so it can point at whatever data config a checkpoint was actually trained against — see `methodology.md` §6) was run against `hardy-lake-19`'s step-300 checkpoint. Filtering out padding (a first pass mistakenly included it — 479 of 492 positions in the sampled sequence were padding, trivially uniform) revealed real, reproducible structure, but **mostly positional, not semantic**, at this early stage: layer 0 routes only the first 1–4 tokens of every sequence to one expert (sequence-start detection) and everything else to another, regardless of topic; layer 6 shows a clean depth-progression through three experts as a sequence gets longer, independent of content; layer 20 was fully collapsed (one expert, ~99% of all real tokens across 5 sampled sequences). One layer (7) showed a plausible content-correlated split — a block of numeric/statistical text routing differently than surrounding narrative prose — but with one sequence and no position control, this could not be distinguished from coincidence. Worth repeating against a later, better-trained checkpoint (§7).

---

## 7. Current State and What Would Be Needed Before Scaling Up

**What this phase established**: the infrastructure-level failure modes are found and fixed — NaN divergence, checkpoint resume, disk exhaustion, and (the big one) the initialization bug that made every prior run's absolute numbers meaningless. One full run now completes cleanly end-to-end with sane losses throughout.

**What this phase has not yet established**, and would be needed before treating this as evidence for a larger/longer training run:

1. **No dense-model baseline at the same budget.** Nothing in this investigation yet shows whether `val_ppl=975` at 20M tokens reflects the MoE architecture working reasonably, or meaningfully underperforming what a much simpler dense model would achieve at the same compute/token budget. This is the single most important missing control before treating the architecture itself as validated.
2. **`n=1`.** Only one run has completed with every fix in place. This project's own history (Phase 1 §3.5: seed-42 vs. seed-123 producing comparably severe collapse onto *different* experts) is a direct precedent for why a single successful run is not yet evidence of a reproducible recipe.
3. **The final near-zero `load_loss` coincides with the learning rate being fully decayed.** It is plausible routing balance was still actively, robustly resolving itself — and equally plausible it simply froze wherever it happened to be as gradients vanished toward the end of the cosine schedule, which would not be evidence of a stable equilibrium at all. Distinguishing these needs either a longer run (does balance hold, or drift, well before the LR floor) or an explicit check of `load_loss`'s trajectory shape near the LR floor specifically.
4. **20M tokens is a smoke-test budget**, not a real pretraining scale. Nothing observed here bounds what happens at 10–100× more tokens — including whether §2's live-refresh mechanism, §3's shared expert, or §6.4's mostly-positional routing structure still hold, degrade, or improve at real scale.

**Recommendation**: not yet ready for a scaled-up run. The right next steps, roughly in order of cost: (a) a same-budget dense-model control run — cheap, and directly answers the most important open question; (b) 1–2 repeat runs of the exact same config with different seeds, to check whether `graceful-universe-21`'s clean finish was the recipe working or a lucky draw; (c) a longer run (~5–10× more tokens) specifically to check whether the end-of-run balance in §6.3 is a real equilibrium or an LR-decay artifact. Only after that evidence exists would scaling up the model size itself be a well-supported decision rather than a hopeful one.

---

## 8. Code Changes Summary

| File | Change |
|---|---|
| `geometry/nystrom.py` | Guarded the `d_alpha_new` division against exact-zero underflow; added live `eps_`/landmark-spectrum refresh in `transform()` (new `_refresh_eps` param, default on); extracted `_fit_landmark_spectrum` shared between `fit()` and the refresh path |
| `geometry/eigensolver.py` | `diffusion_eigenvectors` gained a `random_state` param, seeding ARPACK's starting vector — needed once the eigensolve could rerun mid-training, not just once per landmark set |
| `models/moe_layer.py` | Added mandatory `shared_expert` (unconditional, outside the router); `_compute_diffusion_coords` now also refits when `ndm.landmarks_ is None` (checkpoint-resume fix), not only on the step-count schedule |
| `models/moe_model.py` | Fixed `token_embedding` init (`std=0.02`, was `nn.Embedding` default `std=1.0`); added depth-aware `1/sqrt(2×n_layers)` scaling to every `out_proj`/`down_proj` weight after construction |
| `training/trainer.py` | Added non-finite-loss guard in `train_step` (raises `FloatingPointError` before `backward()`/`optimizer.step()` instead of silently training on NaN/Inf); added `_prune_old_checkpoints` (`keep_last_n_checkpoints`, default 2) |
| `data/dataloader.py` | Fixed `local_data_files` reaching the real Hydra-config-driven training path — a `ConfigAttributeError` several frames deep (`omegaconf.ListConfig` vs. plain `list`), not an obvious error at the call site |
| `scripts/expert_attribution.py` | Added Hydra-style CLI override support (`parse_known_args`), so it can point at the real data config a checkpoint was trained against instead of always falling back to `base_config.yaml`'s defaults |
| `configs/base_config.yaml` | `routing.mu_load`: `0.01` → `2.0` (measured, §6.2); `training.keep_last_n_checkpoints: 2` (new) |

All changes are covered by tests reproducing the specific failure before confirming the fix, not just testing the fix in isolation — e.g. a mixed-batch outlier reproducing the exact NaN-underflow condition; a cross-object checkpoint resume at a step deliberately off the refit schedule, confirmed to crash on the pre-fix code (verified by temporarily reverting the fix) and pass after; a monkeypatched zero-contribution routed path confirming the shared expert's contribution is structurally unconditional; a real disk-bounded checkpoint-pruning scenario; and the random-label cross-entropy check confirming init-time loss lands near the true random baseline (confirmed to fail — 61.5 vs. an expected <13.8 — on the pre-fix code). Full suite: 375 tests passing at last check.

---

## Appendix: Open Questions Carried Forward

- Does the depth-progression / sequence-start routing pattern (§6.4) persist, sharpen into content-sensitivity, or dissolve entirely at a later training step and a larger token budget?
- Should `nu_sep` (currently unchanged at 0.05) be measured the same way `mu_load` was (§6.2)? `sep_loss` showed real instability of its own (swings from -3.5 to -10.6 within ~120 steps in the collapsed run) that was not directly investigated — it may be a downstream symptom of the same collapse dynamic, or a separate miscalibration.
- The self-tuning (per-point adaptive) bandwidth design discussed but not implemented (the "Option 2" alternative to §2.2's global live-refresh) remains a candidate if staleness-driven instability recurs even with the current fix.
