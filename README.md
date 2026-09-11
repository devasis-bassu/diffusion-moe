# Diffusion-Guided Mixture of Experts Transformer

> **Replacing monolithic FFN blocks with geometry-aware specialist networks, routed by diffusion map coordinates computed on post-attention representations.**

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [The Core Idea in Plain English](#the-core-idea-in-plain-english)
3. [Theory Walkthrough](#theory-walkthrough)
   - [Step 1 — Token Embeddings](#step-1--token-embeddings)
   - [Step 2 — Attention as a Single Diffusion Step](#step-2--attention-as-a-single-diffusion-step)
   - [Step 3 — What Attention Does Not Do](#step-3--what-attention-does-not-do)
   - [Step 4 — The FFN as a Semantic Sharpener](#step-4--the-ffn-as-a-semantic-sharpener)
   - [Step 5 — The Cost of Universality](#step-5--the-cost-of-universality)
   - [Step 6 — Why Cosine Distance Is Insufficient](#step-6--why-cosine-distance-is-insufficient)
   - [Step 7 — RoPE: Position Without Contamination](#step-7--rope-position-without-contamination)
   - [Step 8 — Diffusion Maps: Similarity via Random Walks](#step-8--diffusion-maps-similarity-via-random-walks)
   - [Step 9 — The Diffusion Map Embedding](#step-9--the-diffusion-map-embedding)
   - [Step 10 — Nyström: Making It Tractable](#step-10--nyström-making-it-tractable)
   - [Step 11 — Routing to Expert FFNs](#step-11--routing-to-expert-ffns)
   - [Step 12 — Forcing Genuine Specialisation](#step-12--forcing-genuine-specialisation)
   - [Step 13 — The Complete Picture](#step-13--the-complete-picture)
4. [Key Equations](#key-equations)
5. [Architecture](#architecture)
6. [Why This Works: Three Reasons](#why-this-works-three-reasons)
7. [Research Goals](#research-goals)
8. [Experimental Programme](#experimental-programme)
   - [Phase 1 — Geometric Validation](#phase-1--geometric-validation-no-training)
   - [Phase 2 — Controlled Training](#phase-2--controlled-training)
   - [Phase 3 — Benchmarking](#phase-3--benchmarking)
   - [Phase 4 — Specialisation Analysis](#phase-4--specialisation-analysis)
9. [Baselines](#baselines)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Project Setup](#project-setup)
12. [Repository Structure](#repository-structure)
13. [Running Experiments](#running-experiments)
14. [Budget Estimates](#budget-estimates)
15. [References](#references)

---

## What This Is

Standard transformer FFN blocks are monolithic: every token — regardless of meaning — passes through the same enormous weight matrices. With `d_ff = 4d`, the FFN accounts for roughly two-thirds of all parameters in a typical model.

This project proposes replacing that monolithic FFN with a **Mixture of Experts (MoE)** layer where routing is determined by **diffusion map geometry** computed on the post-attention representations. Instead of a learned linear router (Switch Transformer) or a hash-based router (Reformer), each token is routed based on where it sits on the semantic manifold — a coordinate system that captures transitive, multi-hop semantic relationships that cosine similarity alone cannot see.

The key insight is that attention and diffusion geometry are **complementary** rather than redundant:

- Attention uses a single-scale cosine kernel — one hop, one notion of proximity
- Diffusion coordinates use a multi-scale kernel — many hops, transitivity, noise-suppressed
- These capture orthogonal structure; routing on diffusion geometry is genuinely new information

Rotary Position Embedding (RoPE) is integral to this design because it keeps position *out* of the token representations that diffusion geometry operates on — position enters only transiently during the attention score computation and does not contaminate the semantic manifold.

---

## The Core Idea in Plain English

Think of each token as a traveller. Attention is a single step in which the traveller looks around and blends with nearby travellers — they arrive at a context-averaged location. The FFN is then supposed to sharpen that blurry average back to a precise semantic location.

The problem: the FFN has to know about every possible semantic location in all of language. It is a universal sharpener, and universality is expensive.

Our proposal: divide the semantic landscape into regions. Assign a specialist to each region — an expert that only knows how to sharpen representations that land near it. Route each traveller to the right specialist using a map of the semantic landscape derived from how tokens actually flow between each other during a random walk.

That map is a diffusion map. The random walk is the Markov chain induced by the cosine kernel on post-attention representations. The routing is by diffusion distance to expert centroids. The specialists are narrow FFNs, each needing only a fraction of the universal FFN's capacity.

---

## Theory Walkthrough

### Step 1 — Token Embeddings

Text is broken into subword tokens, each mapped to a point in `R^d`:

```
x_i ∈ R^d
```

Tokens with similar meanings sit near each other in this space. But a raw embedding has no context — "bank" near "river" and "bank" near "loan" are the same point.

---

### Step 2 — Attention as a Single Diffusion Step

Attention lets each token look at every other token and ask: *how relevant are you to understanding what I mean right now?* It computes similarity scores, turns them into weights (via softmax), and forms a weighted average:

```
α_ij = softmax( q_i · k_j / √d_k )
z_i  = Σ_j  α_ij · v_j
```

The output `z_i` is a context-aware blend of all tokens, weighted by how much token `i` attends to each `j`. Crucially, the `α_ij` are non-negative and sum to one — they are **transition probabilities**. This is structurally identical to one step of a Markov chain, which is why diffusion geometry is the right framework for what comes next.

---

### Step 3 — What Attention Does Not Do

After the attention step, `z_i` is a convex combination of value vectors. It sits *between* several semantic locations — blurred intentionally to integrate context. But the representation is now soft and diffuse. The model knows what tokens are nearby but has lost precision about exactly where in semantic space this token belongs.

---

### Step 4 — The FFN as a Semantic Sharpener

The FFN receives `z_i` and applies:

```
FFN(z_i) = W₂ σ(W₁ z_i + b₁) + b₂
```

Research by Geva et al. (2021) showed that the rows of `W₁` act as **pattern detectors** — each neuron fires when `z_i` resembles a learned prototype. The nonlinearity `σ` thresholds the response. The columns of `W₂` are **output memories** — what to emit when that pattern fires.

The full layer is:
```
x_i  →[attention]→  z_i  →[FFN]→  h_i
       (blur for           (sharpen for
       context)            precision)
```

Attention smears the point across nearby meanings. The FFN denoises it back to a clean semantic locus.

---

### Step 5 — The Cost of Universality

The FFN width is `d_ff = 4d`. Every token passes through these same weights regardless of its meaning. The width is the price of serving all semantic territory — finance, biology, grammar, narrative — through a single computation. For a 7B model, FFN blocks hold ~4B of the ~7B parameters.

But any individual token only needs the subset of the FFN relevant to its semantic neighbourhood. A financial token does not need the sharpening directions for river semantics. This is the waste we target.

---

### Step 6 — Why Cosine Distance Is Insufficient

The obvious solution: cluster tokens by similarity, assign specialist FFNs to clusters, route by nearest cluster. The problem is that cosine similarity is a single-hop, single-scale notion of proximity.

Two tokens can be strongly related through a chain of intermediate concepts yet look distant under cosine similarity. "Equity" and "dividend" might not be directly close, but are connected through "shareholder", "return", "portfolio". Cosine routing would separate them into different experts, but they need the same sharpening directions.

We need a similarity measure that respects **transitivity** — closeness through chains of connections.

---

### Step 7 — RoPE: Position Without Contamination

Standard positional encoding adds a position vector to each token embedding before any processing begins:

```
x̃_i = x_i + PE(i)          # position permanently mixed into meaning
```

This contaminates every downstream computation with positional information.

RoPE applies a position-dependent rotation only to the query and key vectors, only at the moment of computing attention scores:

```
q̃_m = R(m) W_Q x_m
k̃_n = R(n) W_K x_n
```

Because rotation matrices compose by adding angles, `R(m)ᵀ R(n) = R(n-m)`, so the dot product depends only on the **relative distance** `Δ = n - m`:

```
q̃_m · k̃_n = x_mᵀ W_Qᵀ R(Δ) W_K x_n
```

**Consequence for this project:** position influences which tokens attend to which, but the output `z_i` carries no positional contamination — values `v_j = W_V x_j` are pure content projections. The semantic manifold that diffusion geometry operates on is positionally clean.

---

### Step 8 — Diffusion Maps: Similarity via Random Walks

Imagine a random walker hopping between tokens, where each step goes to a nearby token with probability proportional to their Gaussian similarity. Two tokens are **diffusion-close** if many random walk paths connect them — if they live in the same well-connected semantic cluster.

Build a Markov transition matrix from the post-attention representations:

```
k(z_i, z_j) = exp( -‖z_i - z_j‖² / 2ε² )      # Gaussian kernel

p(z_i, z_j) = k(z_i, z_j) / (q(z_i)^α · q(z_j)^α)   # density normalisation

P_ij = p(z_i, z_j) / Σ_j p(z_i, z_j)           # row-normalise to Markov matrix
```

The `α`-normalisation (Coifman–Lafon) strips out density effects so the geometry reflects manifold structure, not sampling frequency. With `α = 1` we recover the Laplace–Beltrami operator on the underlying manifold.

Diffusion distance after `t` steps:
```
D_t(z_i, z_j)² = Σ_ℓ  λ_ℓ^{2t} (ψ_ℓ(z_i) - ψ_ℓ(z_j))²
```

where `ψ_ℓ` are eigenvectors of `P` and `λ_ℓ` are eigenvalues. The `λ_ℓ^{2t}` weighting **suppresses noise** — small eigenvalues (fast-decaying, local structure) vanish as `t` grows, leaving only coarse semantic organisation.

---

### Step 9 — The Diffusion Map Embedding

Embed each token into a low-dimensional space where Euclidean distance equals diffusion distance:

```
Ψ_t(z_i) = ( λ₁ᵗ ψ₁(z_i),  λ₂ᵗ ψ₂(z_i),  …,  λᵣᵗ ψᵣ(z_i) )  ∈  R^r
```

Only the top `r` eigenvectors are needed — the rest have eigenvalues so small they contribute nothing. The **intrinsic dimension** `r*` is:

```
r* = min{ r :  Σ_{ℓ=1}^r λ_ℓ^{2t}  /  Σ_ℓ λ_ℓ^{2t}  ≥ 0.95 }
```

We compress each token from `R^d` (e.g. d=4096) to `R^{r*}` (expected: r* ~ 10–50), preserving semantic neighbourhood structure. This compact vector is the **semantic address** used for routing.

---

### Step 10 — Nyström: Making It Tractable

Computing the full `n × n` kernel matrix is `O(n²)` — infeasible for long sequences. The Nyström approximation selects `m ≪ n` landmark tokens, computes the exact `m × m` kernel and eigensystem on them, then extends to all tokens cheaply:

```
ψ̃_ℓ(z_i) ≈ (1/λ_ℓ) Σ_{j=1}^m K(z_i, z_j*) ψ_ℓ(z_j*)
```

Cost: `O(m²)` for the landmark eigen-decomposition + `O(nm)` for extension, versus `O(n²)` naively. With `m = 128` landmarks and `n = 4096` tokens, this is a ~32× reduction.

---

### Step 11 — Routing to Expert FFNs

Place `K` expert FFNs, each with a centroid `c_k ∈ R^{r*}` in diffusion space. Route each token based on diffusion distance to centroids:

```
g_k(z_i) = softmax( -‖Ψ_t(z_i) - c_k‖² / τ )
```

Keep only the top-`κ` experts for sparsity (typically κ = 1 or 2):

```
ĝ_k(z_i) = g_k(z_i) · 1[k ∈ top-κ(z_i)]  /  Σ_{j ∈ top-κ} g_j(z_i)
```

The combined output:
```
h_i = Σ_k  ĝ_k(z_i) · Expert_k(z_i)
```

Each expert is a narrower FFN with width `d_ff^{(k)} ≈ d_ff / K`, justified because it only covers a geometrically compact region. Active FLOPs per token: `~κ/K` of the dense baseline.

---

### Step 12 — Forcing Genuine Specialisation

Without any explicit pressure, experts drift toward overlapping coverage. We add a **separation penalty** that pushes expert centroids apart in diffusion space:

```
L_sep = -Σ_{j<k} D_t(c_j, c_k)²
```

Combined with load balancing (penalising uneven token distribution) and task loss:

```
L = L_task  +  μ · L_load  +  ν · L_sep
```

`L_sep` has semantic content, not just engineering value: maximising diffusion distance between centroids forces each expert to own a distinct, non-overlapping region of the semantic manifold. This is the geometric analogue of an orthogonality constraint on basis functions.

**Connection to the spectral gap:** the spectral gap `δ = λ₁ - λ₂` of `P` bounds how many well-separated clusters the manifold supports. Setting `K > 1/δ` produces experts with inherently overlapping coverage — monitor `δ` per layer to guide the choice of `K`.

---

### Step 13 — The Complete Picture

```
x_i                  z_i                  Ψ_t(z_i)          Expert_k           h_i
raw token   →    context-blended   →   semantic address   →   narrow FFN   →   sharp,
no position      positionally clean     r* dimensions         local region      precise
             ↑                     ↑                     ↑
          RoPE: position         Nyström diffusion      routing by
          enters here            map on z_i             diffusion distance
          and exits
```

---

## Key Equations

| Name | Equation |
|------|----------|
| Attention weights | `α_ij = softmax( q_i · k_j / √d_k )` |
| Context vector | `z_i = Σ_j α_ij v_j` |
| RoPE rotation | `q̃_m · k̃_n = x_mᵀ W_Qᵀ R(n−m) W_K x_n` |
| Gaussian kernel | `k(z_i, z_j) = exp(−‖z_i−z_j‖² / 2ε²)` |
| Markov matrix | `P_ij = p(z_i,z_j) / Σ_j p(z_i,z_j)` |
| Diffusion distance | `D_t(z_i,z_j)² = Σ_ℓ λ_ℓ^{2t}(ψ_ℓ(z_i)−ψ_ℓ(z_j))²` |
| Diffusion embedding | `Ψ_t(z_i) = (λ₁ᵗψ₁(z_i), …, λᵣᵗψᵣ(z_i)) ∈ R^r` |
| Intrinsic dimension | `r* = min{r : Σ_{ℓ≤r} λ_ℓ^{2t} / Σ_ℓ λ_ℓ^{2t} ≥ 0.95}` |
| Nyström extension | `ψ̃_ℓ(z_i) ≈ (1/λ_ℓ) Σ_j K(z_i, z_j*) ψ_ℓ(z_j*)` |
| Router | `g_k(z_i) = softmax(−‖Ψ_t(z_i)−c_k‖²/τ)` |
| MoE output | `h_i = Σ_k ĝ_k(z_i) · Expert_k(z_i)` |
| Separation loss | `L_sep = −Σ_{j<k} D_t(c_j,c_k)²` |
| Total loss | `L = L_task + μ L_load + ν L_sep` |

---

## Architecture

```
Input tokens
     │
     ▼
Token Embeddings  (no positional encoding added here)
     │
     ▼  ┌─────────────────────────────────────────────────┐
        │           Transformer Layer × N                  │
        │                                                   │
        │   x_i ──[RMSNorm]──┐                             │
        │                    ▼                             │
        │           RoPE Multi-Head Attention              │
        │           (position via rotation of Q,K only)   │
        │                    │                             │
        │               z_i (context-blended,             │
        │                    positionally clean)           │
        │                    │                             │
        │           ┌────────┴────────┐                   │
        │           │  Nyström        │                    │
        │           │  Diffusion Map  │                    │
        │           │  on z_i         │                    │
        │           └────────┬────────┘                   │
        │                    │                             │
        │               Ψ_t(z_i) ∈ R^{r*}                │
        │                    │                             │
        │              [Router: diffusion                  │
        │               distance to centroids]             │
        │                    │                             │
        │         ┌──────────┴──────────┐                 │
        │    Expert₁  Expert₂  …  Expert_K                │
        │    (narrow  (narrow      (narrow                 │
        │     FFN)     FFN)         FFN)                   │
        │         └──────────┬──────────┘                 │
        │                    │                             │
        │           Weighted aggregate h_i                 │
        │                    │                             │
        │             [RMSNorm + residual]                 │
        └─────────────────────────────────────────────────┘
     │
     ▼
LM Head → logits
```

---

## Why This Works: Three Reasons

**1. The manifold hypothesis is well-supported.**
Token representations at each transformer layer do not fill `R^d` uniformly. They trace a low-dimensional manifold whose intrinsic dimension `r*` is much smaller than `d`. Diffusion maps find this structure; the Phase 1 experiments will quantify it.

**2. RoPE provides a clean separation of concerns.**
Position only enters the computation at the attention score step and immediately exits. The post-attention representations `z_i` that diffusion geometry operates on are semantically pure — no positional contamination to confuse the manifold structure or pollute the clustering.

**3. The architecture extends what the FFN is already doing.**
Geva et al. showed the FFN already functions as a key-value memory with implicit specialisation by pattern type. Our proposal makes that specialisation explicit and geometric — each expert learns sharpening directions appropriate to its local manifold region. We are working with the architecture's natural structure, not against it.

---

## Research Goals

| ID | Goal | Success Criterion |
|----|------|-------------------|
| G1 | Validate manifold hypothesis | `r* ≪ d` confirmed at each layer; post-attention `r*` < pre-attention `r*` |
| G2 | Demonstrate compute reduction | ≥ 40% FFN FLOP reduction at matched task perplexity |
| G3 | Force semantic expert separation | Higher centroid `D_t` and lower routing entropy vs. Switch Transformer baseline |
| G4 | Determine Nyström budget | Minimum `m` for routing-quality coordinates as function of `n` and `r*` |
| G5 | Layer-wise geometry mapping | `r*(l)` and `δ(l)` profile across all layers; identify where MoE is most beneficial |
| G6 | Characterise scaling | How compute savings scale with model size, sequence length, and `K` |

---

## Experimental Programme

### Phase 1 — Geometric Validation (No Training)

**Cost: ~$160 · Duration: 1–2 weeks · Kill switch: if r* ≈ d everywhere, stop.**

Extract post-attention activations from pretrained Llama-2-7B and Mistral-7B across 50K sequences (Wikipedia + C4). For each layer compute:

- **Intrinsic dimension** `r*(l)` via diffusion eigenspectrum energy retention
- **Spectral gap** `δ(l) = λ₁ - λ₂` — how many well-separated clusters the manifold supports
- **Pre vs. post-attention comparison** — test whether attention reduces `r*` (the RoPE prediction)
- **Nyström quality curve** — approximation error vs. `m ∈ {32, 64, 128, 256, 512}`

```bash
python scripts/extract_geometry.py \
    --model mistralai/Mistral-7B-v0.1 \
    --n_sequences 1000 \
    --device cuda
# Results → results/geometry/
```

Expected outputs: `r*(l)` plot, `δ(l)` plot, Nyström error vs. `m` plot, pre/post comparison table.

---

### Phase 2 — Controlled Training

**Cost: ~$24,000 · Duration: 4–8 weeks**

Train all five variants at **300M parameters / 30B tokens** for hyperparameter search, then **1.3B parameters / 100B tokens** for main results.

Hyperparameter sweep at 300M scale (~20 runs, ~$4,800):
- `K ∈ {4, 8, 16}` experts
- `κ ∈ {1, 2}` active experts
- `ν ∈ {0, 0.01, 0.1}` (separation loss weight)
- `μ ∈ {0.01, 0.1}` (load balance weight)

Full training at 1.3B scale (5 variants × ~$2,700 each ≈ $19,000):

```bash
# Dry run to verify pipeline
python scripts/run_sweep.py --dry_run

# Full sweep
python scripts/run_sweep.py

# Full 1.3B training
python scripts/train.py --config-name 1.3b router=diffusion
```

---

### Phase 3 — Benchmarking

**Cost: ~$200 · Duration: 1 week**

Evaluate all five trained variants on:

| Category | Benchmarks |
|----------|-----------|
| Language modelling | Wikitext-103 perplexity, The Pile perplexity, LAMBADA |
| Few-shot reasoning | MMLU, HellaSwag, ARC-Challenge, WinoGrande |
| Long-context | SCROLLS, LongBench |

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/diffusion_moe_1.3b/ \
    --tasks wikitext,lambada,hellaswag,arc_challenge,mmlu
```

---

### Phase 4 — Specialisation Analysis

**Cost: ~$240 · Duration: 1–2 weeks**

This is what distinguishes the paper from a pure efficiency paper.

**Centroid separation over training** — track `D_t(c_j, c_k)` across all expert pairs at each checkpoint. Does separation grow monotonically? Does it collapse?

**Token routing analysis** — for each expert, collect the most-routed token types and compute routing entropy:
```
H_k = -Σ_v  p_{kv} log p_{kv}
```
Lower entropy = crisper specialisation. Compare across all routing methods.

**Probing classifiers** — train linear probes on each expert's output to predict POS tag, NER label, syntactic dependency role, semantic similarity cluster. Expert-specific probe accuracy measures specialisation quality.

**Dead expert analysis** — track coefficient of variation of token counts across experts:
```
CV = std(n₁, …, n_K) / mean(n₁, …, n_K)
```

---

## Baselines

All baselines share identical `d_model`, `n_layers`, `n_heads`, and total token budget. Only the FFN routing mechanism differs.

| Label | Router | Purpose |
|-------|--------|---------|
| **Dense** | None — full FFN | Primary compute comparison |
| **Random-MoE** | Uniform random assignment | Tests whether routing matters at all |
| **Switch-MoE** | Linear projection + softmax | Standard learned router comparison |
| **Cosine-MoE** | Cosine similarity in R^d | Isolates multi-hop geometry contribution |
| **Diffusion-MoE** | Diffusion distance in R^{r*} | This project's contribution |

The critical comparison is **Diffusion-MoE vs. Cosine-MoE** — this directly validates whether the multi-scale, transitive geometry of diffusion maps provides anything beyond single-step cosine routing.

Switch config:
```yaml
router: diffusion    # options: diffusion | cosine | switch | random
```

---

## Evaluation Metrics

### Geometric Validity
- `r*(l)` per layer — should be ≪ d, expected 10–50
- Spectral gap `δ(l)` — guides expert count selection per layer
- Nyström error at various `m` — target < 5% at `m = 128`
- Routing stability — fraction of tokens changing expert across batches

### Compute Efficiency
- Active FLOPs per token — target κ/K fraction of dense baseline
- Wall-clock throughput (tokens/sec)
- Peak GPU memory
- Nyström overhead as % of forward pass

### Task Performance
- Perplexity (Wikitext-103, The Pile) — primary regression signal
- MMLU, HellaSwag, ARC accuracy — within 1–2% of dense baseline
- LAMBADA, SCROLLS, LongBench — long-range stability

### Expert Specialisation
- Mean pairwise centroid `D_t` — higher is better
- Routing entropy `H_k` per expert — lower is crisper
- Expert token CV — target < 0.3 (load balanced)
- Linear probe accuracy per expert per linguistic category
- Comparison of all above vs. Switch-MoE baseline

---

## Project Setup

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) or `pyenv` |
| CUDA Toolkit | 12.1+ | [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| VS Code | Latest | [code.visualstudio.com](https://code.visualstudio.com) |
| Claude Code | Latest | `npm install -g @anthropic/claude-code` |
| uv | Latest | `pip install uv` |

### Installation

```bash
git clone https://github.com/devasis-bassu/diffusion-moe
cd diffusion-moe

# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
make install

# Copy and fill in environment variables
cp .env.example .env
# Edit .env: add WANDB_API_KEY, HF_TOKEN, DATA_DIR
```

### VS Code Extensions

Install from the Extensions panel (`Ctrl+Shift+X`):
- `ms-python.python` — Python language support
- `ms-python.vscode-pylance` — type checking
- `charliermarsh.ruff` — linting and formatting
- `eamodio.gitlens` — git history

### Start Claude Code

```bash
# In the VS Code integrated terminal
claude
```

Paste the prompts from the [Claude Code Project Guide PDF](docs/diffusion_moe_project_guide.pdf) in order.

---

## Repository Structure

```
diffusion-moe/
├── src/diffusion_moe/
│   ├── data/
│   │   ├── tokenizer.py          # TokenizerWrapper (HuggingFace)
│   │   ├── dataset.py            # StreamingTextDataset
│   │   └── dataloader.py         # build_dataloaders()
│   ├── models/
│   │   ├── rope.py               # RotaryEmbedding
│   │   ├── attention.py          # RoPEMultiHeadAttention
│   │   ├── ffn.py                # FeedForward (SwiGLU)
│   │   ├── expert_ffn.py         # ExpertFFN (narrow variant)
│   │   ├── transformer_block.py  # TransformerBlock
│   │   ├── base_model.py         # BaseTransformer (dense baseline)
│   │   ├── moe_layer.py          # DiffusionMoELayer
│   │   └── moe_model.py          # DiffusionMoETransformer
│   ├── geometry/
│   │   ├── kernel.py             # gaussian_kernel(), bandwidth_median_heuristic()
│   │   ├── markov.py             # coifman_lafon_normalise()
│   │   ├── eigensolver.py        # diffusion_eigenvectors()
│   │   ├── nystrom.py            # NystromDiffusionMap
│   │   └── intrinsic_dim.py      # estimate_intrinsic_dim(), spectral_gap()
│   ├── routing/
│   │   ├── centroids.py          # ExpertCentroids (nn.Parameter)
│   │   ├── router.py             # DiffusionRouter
│   │   ├── load_balance.py       # coefficient_of_variation_loss()
│   │   └── separation.py        # centroid_separation_loss()
│   ├── training/
│   │   ├── losses.py             # total_loss()
│   │   ├── optimizer.py          # build_optimizer(), build_scheduler()
│   │   └── trainer.py            # Trainer class
│   ├── evaluation/
│   │   ├── perplexity.py         # compute_perplexity()
│   │   ├── benchmarks.py         # run_lm_eval()
│   │   └── routing_analysis.py   # RoutingAnalyser
│   └── utils/
│       ├── logging.py
│       ├── checkpointing.py
│       └── profiling.py
├── configs/
│   ├── base_config.yaml
│   ├── model/
│   │   ├── 300m.yaml
│   │   └── 1.3b.yaml
│   └── experiment/
│       ├── sweep_300m.yaml
│       └── full_1.3b.yaml
├── scripts/
│   ├── train.py                  # Main training entry point
│   ├── evaluate.py               # Evaluation entry point
│   ├── extract_geometry.py       # Phase 1 geometry validation
│   └── run_sweep.py              # W&B hyperparameter sweep
├── tests/
│   ├── data/
│   ├── models/
│   ├── geometry/
│   ├── routing/
│   └── test_integration.py
├── notebooks/
│   ├── geometry_exploration.ipynb
│   └── routing_analysis.ipynb
├── results/
│   ├── geometry/
│   └── eval/
├── docs/
│   └── diffusion_moe_project_guide.pdf
├── pyproject.toml
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Running Experiments

```bash
# Run test suite
make test

# Phase 1: geometry validation (no training — run this first)
python scripts/extract_geometry.py --model mistralai/Mistral-7B-v0.1

# Phase 2: dry run (verifies pipeline end-to-end, 100 steps)
python scripts/run_sweep.py --dry_run

# Phase 2: full hyperparameter sweep at 300M scale
python scripts/run_sweep.py

# Phase 2: full 1.3B training (best config from sweep)
python scripts/train.py --config-name 1.3b router=diffusion

# Evaluate a checkpoint
python scripts/evaluate.py \
    --checkpoint checkpoints/diffusion_moe_1.3b/ \
    --tasks wikitext,hellaswag,arc_challenge,mmlu

# Compare two checkpoints side by side
python scripts/evaluate.py \
    --checkpoint checkpoints/diffusion_moe_1.3b/ \
    --compare checkpoints/switch_moe_1.3b/
```

---

## Budget Estimates

| Phase | Description | Estimated Cost |
|-------|-------------|---------------|
| Phase 1 | Geometry validation (no training) | ~$160 |
| Phase 2 | 300M sweep + 1.3B training (5 variants) | ~$24,000 |
| Phase 3 | Benchmarking (inference only) | ~$200 |
| Phase 4 | Specialisation analysis | ~$240 |
| **Core total** | | **~$24,600** |
| Optional: 7B scale-up (Dense + Diffusion-MoE only) | | ~$40,000 |
| **Full paper budget** | | **~$65,000** |

**Decision gate at ~$7,000:** Phase 1 + Phase 2 sweeps + two 1.3B runs. Enough to know whether the project warrants full investment.

**Cost reduction options:**
- GCP TPU Research Credits (~40% cheaper than H100 on-demand for training)
- Drop Random-MoE baseline (its result is predictable) → saves ~$5,500
- Frame 1.3B as the main result, 7B as future work

---

## References

### Transformer Architecture
- Vaswani et al. (2017). *Attention is All You Need.* NeurIPS.
- Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS.
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.* NeurIPS. (Chinchilla)

### Rotary Position Embedding
- Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* Neurocomputing. (arXiv:2104.09864)
- Touvron et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288.
- Peng et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models.* arXiv:2309.00071.

### FFN as Pattern Detectors
- Geva et al. (2021). *Transformer Feed-Forward Layers are Key-Value Memories.* EMNLP.
- Geva et al. (2022). *Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space.* EMNLP.
- Meng et al. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS.

### Mixture of Experts
- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR.
- Fedus et al. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR.
- Zhou et al. (2022). *Mixture-of-Experts with Expert Choice Routing.* NeurIPS.
- Jiang et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Chi et al. (2022). *On the Representation Collapse of Sparse Mixture of Experts.* NeurIPS.

### Diffusion Map Geometry
- Coifman & Lafon (2006). *Diffusion Maps.* Applied and Computational Harmonic Analysis.
- Coifman et al. (2005). *Geometric diffusions as a tool for harmonic analysis and structure definition of data.* PNAS.
- Belkin & Niyogi (2003). *Laplacian Eigenmaps for Dimensionality Reduction.* Neural Computation.
- Williams & Seeger (2001). *Using the Nyström Method to Speed Up Kernel Machines.* NeurIPS.
- Drineas & Mahoney (2005). *On the Nyström Method for Approximating a Gram Matrix.* JMLR.

### Related Efficient Attention
- Kitaev et al. (2020). *Reformer: The Efficient Transformer.* ICLR.
- Choromanski et al. (2021). *Rethinking Attention with Performers.* ICLR.
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention.* NeurIPS.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{diffusion-moe-2026,
  title  = {Diffusion-Guided Mixture of Experts for Feed-Forward
             Network Compression in Large Language Models},
  year   = {2026},
  note   = {Research in progress. \url{https://github.com/devasis-bassu/diffusion-moe}}
}
```