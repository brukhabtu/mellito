"""Modal app for the P7 Ornith bf16 PREFERENCE fit — a separate app
(`ornith-pref`) so it never disturbs the deployed `ornith-harness` serving stack
or the P4 `ornith-lora` app. It reuses the P4-proven machinery in
train_lora.py / chat_template_adapt.py almost verbatim; the only new thing is the
loss.

===========================================================================
METHOD DECISION (Option B: weighted-CE unlikelihood on the proven SFTTrainer)
===========================================================================
The goal is a POSITIVE gradient on ~96 verifier-passing trajectories and a
NEGATIVE gradient on ~173 failing ones (labels carried in the data), with the
HARD REQUIREMENT that loss touches ASSISTANT tokens only — the
`{% generation %}` spans — never tool_result / system / user content, for
positives OR negatives. Option A (TRL `KTOTrainer`) is REJECTED, by design not
by hope: KTO consumes (prompt, completion, label) triples and computes its loss
over the whole completion; it has no documented, version-pinned mechanism to
confine gradient to the assistant-only `{% generation %}` sub-spans of a
multi-turn completion whose interleaved tool_result turns MUST stay gradient-free.
Since I cannot GUARANTEE the masking from behaviour I control, the pre-approved
fallback (Option B) is the correct call.

Option B subclasses the P4-PROVEN `SFTTrainer` and overrides only `compute_loss`.
`assistant_only_loss=True` does the masking EXACTLY as proven in P4 (the collator
turns the `{% generation %}` `assistant_masks` into `labels` with -100 on every
non-assistant token), so both the positive and the negative gradient are confined
to assistant tokens by construction. Per-row: passing rows use standard token
cross-entropy (raise p of good tokens); failing rows use the BOUNDED unlikelihood
term −log(1 − p) summed over their assistant tokens (lower p of bad tokens). We
prefer unlikelihood over the naive `CE(pos) − λ·CE(neg)` form because −λ·CE is
unbounded below — as p→0 it drives the loss to −∞ and the gradient explodes —
whereas −log(1 − p) is bounded below by 0, is minimised at the finite target
p→0, and only diverges toward the thing we are minimising away from (p→1), which
we additionally clamp. The negative term is scaled by
`neg_lambda·(n_pos/n_neg)` (default neg_lambda=0.2), which both downweights the
push-away and cancels the ~2:1 pass/fail COUNT imbalance so negatives cannot
dominate; combined with per-row mean-over-assistant-tokens normalisation, long
trajectories don't dominate short ones either. lr is HALF of P4's (5e-5): we are
pushing away as well as toward, so we stay conservative.

WHY the label survives TRL's preprocessing (verified, not hoped): against the
PINNED trl==0.29.1 source, `SFTTrainer._prepare_dataset`'s tokenisation `.map`
passes NO `remove_columns` that targets `label` (only `conversations`/`messages`
are dropped), and `assistant_only_loss=True` produces the `assistant_masks`
column consumed by `DataCollatorForLanguageModeling`. The base HF Trainer would
still strip `label` via `remove_unused_columns` (SFTTrainer's signature columns
are a FIXED list that excludes it), so we set `remove_unused_columns=False` and
wrap the proven collator: our wrapper pops `label` off each row BEFORE the base
collator runs (so it only ever sees the tensor keys it knows) and re-attaches a
per-row `pref_is_pos` flag that `compute_loss` reads. P4 ran these libraries
UNPINNED; we PIN them here for reproducibility — the pins are the only combo that
loads Ornith (its remote code hard-requires transformers>=5.8.1) with the current
stable TRL whose masking + column behaviour we verified.

Structural note (same as train_lora.py): this file is built to import cleanly and
be structurally correct; the GPU/Modal bodies are NOT exercised in the harness.
The proof they run is `preflight_pref` on Modal (CPU, tokenizer-only) BEFORE any
H200 boot.
"""

import os

import modal

# Default location of the preference JSONL produced by export_preferences.py. It
# is gitignored/ephemeral (the scratchpad), so it is added to the images at build
# time from PREF_PATH; override with `PREF_PATH=/abs/path modal run ...`.
_DEFAULT_PREF_PATH = (
    "/tmp/claude-0/-home-user-mellito/"
    "5ffb3023-5238-5cf4-a306-ed31d13a97c9/scratchpad/pref.jsonl"
)
PREF_PATH = os.environ.get("PREF_PATH", _DEFAULT_PREF_PATH)

MODEL = "deepreinforce-ai/Ornith-1.0-35B"  # bf16 repo (NOT the -FP8 serving one)
GPU = "H200"  # same lever as P4: 67GB frozen bf16 base + 32k-seq training
MINUTES = 60

app = modal.App("ornith-pref")

# Reuse the SAME bf16 base weights volume P4 populated (download step lives in
# train_lora.py::download_bf16_weights) and the SAME adapters volume — we write a
# `pref-<ts>` adapter alongside the P4 `lora-*` ones, never overwriting them.
weights_bf16 = modal.Volume.from_name("ornith-weights-bf16", create_if_missing=True)
adapters = modal.Volume.from_name("ornith-adapters", create_if_missing=True)

# HF token only needed if the repo turns out gated; attach nothing by default
# (same rationale as train_lora.py::_HF_SECRETS).
_HF_SECRETS: list = []

# --- pinned dependency stack (P4 ran UNPINNED; pinned here for reproducibility) -
# Load-bearing pins:
#   * transformers==5.8.1 — the EXACT floor Ornith-1.0-35B's remote code declares
#     ("Transformers >= 5.8.1"); older lines will not load the hybrid Mamba/GDN
#     arch. Satisfies trl 0.29.1's transformers>=4.56.2.
#   * trl==0.29.1 — current stable; its SFTTrainer._prepare_dataset was VERIFIED
#     to (a) keep an extra `label` column through tokenisation and (b) do
#     assistant-only masking via {% generation %} -> assistant_masks -> labels.
#   * peft==0.19.1 — >=0.18 is required for the transformers-v5 line.
# torch is a FLOOR (not exact) so the CUDA/Hopper wheel resolves against
# transformers 5.8.1 without an over-constrained solve. datasets/accelerate are
# floored to trl 0.29.1's declared minima.
_TRL = "trl==0.29.1"
_TRANSFORMERS = "transformers==5.8.1"
_PEFT = "peft==0.19.1"
_TORCH = "torch>=2.7"
_DATASETS = "datasets>=4.7.0"
_ACCELERATE = "accelerate>=1.4.0"

_TEMPLATE_LOCAL = os.path.join(os.path.dirname(__file__), "ornith_chat_template.jinja")


# --- images ----------------------------------------------------------------

# Preflight needs the real tokenizer + jinja + our pure helpers + a CPU torch (to
# exercise the custom loss on a tiny synthetic batch), but NO GPU and no
# trl/peft/datasets. The preference JSONL is baked in at /data/pref.jsonl so the
# gate renders REAL rows.
preflight_pref_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(_TRANSFORMERS, _TORCH, "jinja2", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("chat_template_adapt", "export_trajectories")
    .add_local_file(PREF_PATH, "/data/pref.jsonl")
    .add_local_file(_TEMPLATE_LOCAL, "/ornith_chat_template.jinja")
)

# Training image: the pinned TRL+PEFT stack (the arch-agnostic path). As in P4 we
# do NOT bake the fragile unsloth wheel in — train_pref tries `import unsloth` at
# runtime and falls straight to TRL+PEFT if it is absent / can't load the arch.
# The preference JSONL is baked in at /data/pref.jsonl.
train_pref_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        _TORCH,
        _TRANSFORMERS,
        _TRL,
        _PEFT,
        _DATASETS,
        _ACCELERATE,
        "bitsandbytes",
        "jinja2",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          # reduce allocator fragmentation on the memory-tight training GPU
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("chat_template_adapt", "export_trajectories")
    .add_local_file(PREF_PATH, "/data/pref.jsonl")
    .add_local_file(_TEMPLATE_LOCAL, "/ornith_chat_template.jinja")
)


# --- pure helpers (no torch / modal at import time) -------------------------

def neg_weight_from_counts(n_pos, n_neg, neg_lambda):
    """Scale applied to the negative unlikelihood term.

    Returns `neg_lambda * (n_pos / n_neg)` so that, summed over an epoch, the
    negative gradient mass is exactly `neg_lambda` times the positive mass,
    REGARDLESS of the ~96 pos vs ~173 neg count imbalance: each positive row
    contributes mean-token CE (weight 1) and each negative row contributes
    `neg_lambda*(n_pos/n_neg)*mean-token UL`, so Σ_neg ≈ neg_lambda·n_pos·mean(UL)
    matches Σ_pos ≈ n_pos·mean(CE) up to the neg_lambda ratio. n_neg==0 -> 0.0
    (degenerates to pure SFT on the positives; caller warns).
    """
    if n_neg <= 0:
        return 0.0
    return float(neg_lambda) * (float(n_pos) / float(n_neg))


def template_token_len(tok_out) -> int:
    """Token count from apply_chat_template(tokenize=True) output, robust to the
    return-shape change across transformers versions: 5.x returns a dict-like
    ({input_ids, attention_mask} — len() of it is its KEY COUNT, the bug the
    preflight caught printing p50=2), older versions a flat id list, batched
    calls a list-of-lists."""
    ids = tok_out["input_ids"] if hasattr(tok_out, "keys") else tok_out
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def _length_stats(lengths, max_length):
    """p50/p90/max + count-over-max_length for a list of token lengths (pure;
    nearest-rank percentiles, no numpy)."""
    import math
    xs = sorted(int(x) for x in lengths)
    n = len(xs)

    def pct(q):
        if n == 0:
            return 0
        k = max(0, min(n - 1, int(math.ceil(q * n)) - 1))
        return xs[k]

    return {
        "count": n,
        "p50": pct(0.5),
        "p90": pct(0.9),
        "max": xs[-1] if n else 0,
        "over_max_length": sum(1 for x in xs if x > max_length),
    }


def pref_loss_from_logits(logits, labels, is_pos, neg_weight, neg_p_clamp=1e-4):
    """Per-row preference loss on ASSISTANT tokens only (torch, imported lazily).

    Args:
      logits: (B, T, V) raw LM logits.
      labels: (B, T) LM labels with -100 on every non-assistant token — this is
              the assistant_only_loss mask (set by the proven collator from the
              `{% generation %}` assistant_masks). It is the ONE thing that keeps
              BOTH the positive and the negative gradient off tool_result /
              system / user tokens.
      is_pos: (B,) 1.0 for verifier-passing rows, 0.0 for failing rows.
      neg_weight: scalar for the negative unlikelihood term (already folds
              neg_lambda AND the pos/neg class-balance factor; see
              neg_weight_from_counts).
      neg_p_clamp: p is clamped to <= 1 - neg_p_clamp so −log(1 − p) can't diverge.

    Positive rows: token cross-entropy −log p (raise p of good tokens).
    Negative rows: bounded unlikelihood −log(1 − p) (lower p of bad tokens).
    Each row is normalised by its own assistant-token count (mean); the batch
    loss is the mean over rows. Returns a scalar tensor.
    """
    import torch
    import torch.nn.functional as F

    if logits.dim() != 3:
        raise ValueError(f"expected logits (B,T,V), got {tuple(logits.shape)}")
    b, t, v = logits.shape
    # Causal shift: position i predicts token i+1. MEMORY-CRITICAL (learned the
    # hard way — first C1 run OOM'd at step 17/19 trying to allocate a 23.7GiB
    # fp32 full-sequence logit copy): select the assistant positions FIRST,
    # upcast only those to fp32, and compute CE in bounded chunks. transformers'
    # own loss survives 32k because it is fused/chunked internally; a naive
    # full-tensor .float() is not. Same math, ~10x lower peak.
    shift_labels = labels[:, 1:].contiguous()           # (B, T-1); -100 = non-assistant
    is_pos = is_pos.to(logits.device).view(-1)
    chunk = 2048                                        # bounded fp32 transient per step

    rows = []
    for i in range(b):
        lab_i = shift_labels[i]
        m_i = lab_i.ne(-100)
        if int(m_i.sum()) == 0:
            rows.append(logits.new_zeros(()))            # no assistant tokens -> 0
            continue
        ce_parts = []
        for s in range(0, t - 1, chunk):
            e = min(s + chunk, t - 1)
            m_c = m_i[s:e]
            if not bool(m_c.any()):
                continue
            # fp32 only for the ASSISTANT positions of this chunk (n_c, V)
            lg_c = logits[i, s:e, :][m_c].float()
            ce_parts.append(F.cross_entropy(lg_c, lab_i[s:e][m_c],
                                            reduction="none"))
        ce_i = torch.cat(ce_parts)                       # (n,) = −log p(true tok)
        if bool(is_pos[i] > 0.5):
            rows.append(ce_i.mean())                     # positive: token CE
        else:
            p = torch.exp(-ce_i).clamp(max=1.0 - neg_p_clamp)  # p(true tok), guarded
            rows.append(neg_weight * (-torch.log1p(-p)).mean())  # negative: −log(1−p)
    return torch.stack(rows).mean()


# --- preflight the preference data + mask + loss (CPU, no GPU) ---------------

@app.function(image=preflight_pref_image, timeout=15 * MINUTES, secrets=_HF_SECRETS)
def preflight_pref(n_per_label=3, max_length=32768):
    """Pre-GPU gate for the preference fit. Run:  modal run infra/train_pref.py::preflight_pref

    Pulls ONLY the tokenizer + template (a few MB, no weights, no GPU), then for
    real rows of /data/pref.jsonl of BOTH labels:
      - renders through the real tokenizer with return_assistant_tokens_mask and
        asserts the mask is NON-EMPTY and covers ONLY assistant spans (no
        <tool_response> / <|im_start|>user / <|im_start|>system text inside any
        masked span; a <think> assistant marker present inside the mask),
      - prints per-label token-length stats (p50/p90/max, count > max_length),
    and finally exercises one forward+backward of the custom loss on a tiny
    synthetic batch, asserting it is finite AND that gradient is ZERO on every
    non-assistant (label==-100) position — the CPU proof that the loss confines
    both positive and negative gradient to assistant tokens. Raises (FAIL) on any
    failed assertion so a bad state surfaces BEFORE any H200 spend.
    """
    import json

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    import chat_template_adapt as cta

    snapshot_download(
        MODEL,
        allow_patterns=["tokenizer*", "*.jinja", "chat_template*",
                        "special_tokens_map.json", "config.json"],
        local_dir="/tok",
    )
    tok = AutoTokenizer.from_pretrained("/tok", trust_remote_code=True)

    # --- read real preference rows, bucketed by label -----------------------
    by_label = {"pass": [], "fail": []}
    with open("/data/pref.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("label") in by_label:
                by_label[ex["label"]].append(ex)
    for lbl in ("pass", "fail"):
        if not by_label[lbl]:
            raise SystemExit(
                f"preflight FAIL: no '{lbl}' examples in /data/pref.jsonl")

    # --- resolve the assistant_only_loss template (marker install / byte check) -
    stock = tok.chat_template
    if stock is None:
        raise SystemExit("preflight FAIL: tokenizer has no chat_template")
    had_markers = cta.has_generation_markers(stock)
    if had_markers:
        patched = stock
        print("[preflight] stock template already has generation markers")
    else:
        with open("/ornith_chat_template.jinja") as fh:
            patched = fh.read()
        probe = by_label["pass"][0]
        m0 = cta.anthropic_to_template_messages(probe["messages"])
        tok.chat_template = stock
        s_txt = tok.apply_chat_template(m0, tokenize=False)
        tok.chat_template = patched
        p_txt = tok.apply_chat_template(m0, tokenize=False)
        if s_txt != p_txt:
            raise SystemExit(
                "preflight FAIL: committed patched template does not render "
                "byte-identical to the model's stock template — the upstream "
                "chat_template changed; re-patch infra/ornith_chat_template.jinja")
        print("[preflight] committed patched template renders byte-identical to "
              "stock; generation markers added")

    # --- per-label token-length stats over ALL rows (cheap tokenise-only) ----
    for lbl in ("pass", "fail"):
        lengths = []
        for ex in by_label[lbl]:
            out = tok.apply_chat_template(
                cta.anthropic_to_template_messages(ex["messages"]), tokenize=True)
            lengths.append(template_token_len(out))
        st = _length_stats(lengths, max_length)
        print(f"[preflight] label={lbl}: n={st['count']} p50={st['p50']} "
              f"p90={st['p90']} max={st['max']} "
              f"over_{max_length}={st['over_max_length']}")

    # --- assistant-only mask verification on N rows per label ----------------
    all_ok = True
    for lbl in ("pass", "fail"):
        for j, ex in enumerate(by_label[lbl][:n_per_label]):
            rep = cta.render_and_verify(tok, patched, ex)
            enc = tok.apply_chat_template(
                cta.anthropic_to_template_messages(ex["messages"]),
                tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
            ids = enc["input_ids"]
            masks = enc.get("assistant_masks")
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if masks and isinstance(masks[0], list):
                masks = masks[0]
            masks = masks or [0] * len(ids)

            leak = None
            for (s, e) in rep["assistant_mask_spans"]:
                span_txt = tok.decode(ids[s:e])
                if ("<tool_response>" in span_txt
                        or "<|im_start|>user" in span_txt
                        or "<|im_start|>system" in span_txt):
                    leak = span_txt[:160]
                    break
            masked_txt = tok.decode(
                [ids[i] for i in range(len(ids)) if i < len(masks) and masks[i]])
            checks = {
                "mask_nonempty": rep["n_assistant_tokens"] > 0,
                "no_nonassistant_in_mask": leak is None,
                "assistant_marker_present": "<think>" in masked_txt,
            }
            ok = all(checks.values())
            all_ok = all_ok and ok
            print(f"[preflight] {lbl} sample {j} ({ex.get('task')}): "
                  f"{'PASS' if ok else 'FAIL'} checks={checks} "
                  f"n_assistant_tokens={rep['n_assistant_tokens']} "
                  f"n_tokens={rep['n_tokens']}")
            if leak is not None:
                print(f"    LEAK: non-assistant text inside mask span: {leak!r}")

    # --- exercise the custom loss on a tiny synthetic batch (CPU) ------------
    import torch

    torch.manual_seed(0)
    b, t, v = 2, 6, 8
    logits = torch.randn(b, t, v, requires_grad=True)
    labels = torch.full((b, t), -100, dtype=torch.long)
    labels[0, 3:] = torch.randint(0, v, (t - 3,))   # assistant tokens on the tail
    labels[1, 3:] = torch.randint(0, v, (t - 3,))
    is_pos = torch.tensor([1.0, 0.0])               # row0 pass, row1 fail
    nw = neg_weight_from_counts(96, 173, 0.2)
    loss = pref_loss_from_logits(logits, labels, is_pos, nw)
    if not torch.isfinite(loss):
        raise SystemExit("preflight FAIL: custom loss is not finite")
    loss.backward()
    g = logits.grad
    if g is None or not torch.isfinite(g).all():
        raise SystemExit("preflight FAIL: custom-loss gradient is not finite")
    # Gradient MUST be zero wherever the (shifted) label is -100 — i.e. on every
    # non-assistant position. logits[:, i, :] feeds the prediction of token i+1,
    # so it is masked iff labels[:, i+1] == -100.
    shift_valid = labels[:, 1:].ne(-100)            # (B, T-1)
    grad_leak = False
    for i in range(b):
        for pos in range(t - 1):
            if not bool(shift_valid[i, pos]) and int(torch.count_nonzero(g[i, pos])) != 0:
                grad_leak = True
    if grad_leak:
        all_ok = False
        print("[preflight] FAIL: gradient leaked onto a non-assistant position")
    else:
        print(f"[preflight] custom-loss synthetic forward/backward OK: "
              f"loss={float(loss):.4f} neg_weight={nw:.4f}; gradient confined to "
              "assistant tokens (non-assistant positions have zero grad)")

    print(f"\n[preflight] {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit("preflight FAIL: see per-check output above")
    return {"had_markers": had_markers,
            "n_pass": len(by_label["pass"]), "n_fail": len(by_label["fail"]),
            "checked_per_label": min(n_per_label, len(by_label["pass"]),
                                     len(by_label["fail"]))}


# --- the preference LoRA fit -----------------------------------------------

@app.function(image=train_pref_image,
              volumes={"/weights_bf16": weights_bf16, "/adapters": adapters},
              gpu=GPU, timeout=180 * MINUTES, secrets=_HF_SECRETS)
def train_pref(pref_path_in_image="/data/pref.jsonl", epochs=1, lr=5e-5,
               max_length=32768, neg_lambda=0.2, run_id=None):
    """Fit a bf16 preference LoRA on the pass/fail trajectory set on one H200.

    Unsloth-primary / TRL+PEFT-fallback (chosen at runtime). Serializes turns
    through the model's own qwen3_xml template patched for assistant_only_loss,
    length-FILTERS overlong examples per label (never truncates), and trains with
    a custom compute_loss: token-CE on `pass` rows, bounded unlikelihood on `fail`
    rows (scaled by neg_lambda·n_pos/n_neg). Saves adapter + tokenizer + patched
    template + run_meta.json into the shared adapters Volume as pref-<UTC ts>.
    """
    import glob
    import json
    import time

    import torch
    from datasets import load_dataset

    import chat_template_adapt as cta

    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    run_id = run_id or f"pref-{ts}"
    out_dir = f"/adapters/{run_id}"
    weights_dir = "/weights_bf16"
    if not glob.glob(os.path.join(weights_dir, "*.safetensors")):
        raise SystemExit(
            f"no bf16 weights in {weights_dir} — run "
            "`modal run infra/train_lora.py::main --step download` first "
            "(train_pref reuses the P4 weights volume)")

    # --- load base model + tokenizer: Unsloth first, eager HF fallback -------
    # (verbatim from train_lora.py::train_lora — the arch-load path is proven.)
    using_unsloth = False
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=weights_dir,
            max_seq_length=max_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,  # bf16 LoRA, not QLoRA
            trust_remote_code=True,
        )
        using_unsloth = True
        print("[train] loaded base via Unsloth FastLanguageModel")
    except Exception as e:  # Unsloth unsupported for this custom arch -> eager
        print(f"[train] Unsloth path unavailable ({e}); falling back to "
              "AutoModelForCausalLM eager")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            weights_dir, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                weights_dir, torch_dtype=torch.bfloat16,
                trust_remote_code=True, attn_implementation="sdpa")
        except (ValueError, Exception) as _sdpa_err:  # noqa: BLE001
            print(f"[train] sdpa unavailable ({_sdpa_err}); using eager")
            model = AutoModelForCausalLM.from_pretrained(
                weights_dir, torch_dtype=torch.bfloat16,
                trust_remote_code=True, attn_implementation="eager")

    # --- install the patched chat template (assistant_only_loss markers) -----
    # Committed, preflight-verified patched template (byte-identical to stock,
    # adds only the {% generation %} markers). Setting it BEFORE the trainer means
    # TRL sees markers already present and does not re-patch.
    stock = tokenizer.chat_template
    if stock is None:
        raise SystemExit("train FAIL: tokenizer has no chat_template")
    if cta.has_generation_markers(stock):
        template = stock
    else:
        with open("/ornith_chat_template.jinja") as fh:
            template = fh.read()
        print("[train] installed committed patched chat_template "
              "(generation markers for assistant_only_loss)")
    tokenizer.chat_template = template

    # --- load + shape the preference set (KEEP the label through the map) ----
    ds = load_dataset("json", data_files=pref_path_in_image, split="train")

    def _shape(ex):
        return {"messages": cta.anthropic_to_template_messages(ex["messages"]),
                "label": ex["label"]}

    ds = ds.map(_shape, remove_columns=[c for c in ds.column_names
                                        if c not in ("messages", "label")])

    # Length-FILTER per label (never truncate): a truncated trajectory would teach
    # a broken tail. Count drops per label so run_meta is honest and neg_weight is
    # computed on the KEPT counts.
    def _ntok(ex):
        out = tokenizer.apply_chat_template(ex["messages"], tokenize=True)
        return {"_ntok": template_token_len(out)}

    ds = ds.map(_ntok)
    labels_all, ntoks_all = ds["label"], ds["_ntok"]
    n_pos_all = sum(1 for lbl in labels_all if lbl == "pass")
    n_neg_all = sum(1 for lbl in labels_all if lbl == "fail")
    n_filt_pos = sum(1 for lbl, n in zip(labels_all, ntoks_all)
                     if lbl == "pass" and n > max_length)
    n_filt_neg = sum(1 for lbl, n in zip(labels_all, ntoks_all)
                     if lbl == "fail" and n > max_length)
    ds = ds.filter(lambda ex: ex["_ntok"] <= max_length).remove_columns("_ntok")
    n_pos = n_pos_all - n_filt_pos
    n_neg = n_neg_all - n_filt_neg
    print(f"[train] length-filter kept pos={n_pos}/{n_pos_all} "
          f"(dropped {n_filt_pos}) neg={n_neg}/{n_neg_all} (dropped {n_filt_neg}) "
          f"over {max_length} tokens")
    if n_pos == 0:
        raise SystemExit("train FAIL: no positive (pass) rows survive the "
                         "length filter — refusing to train a push-away-only fit")
    if n_neg == 0:
        print("[train] WARNING: no negative (fail) rows survive — this degenerates "
              "to pure SFT on positives (neg_weight=0)")

    neg_weight = neg_weight_from_counts(n_pos, n_neg, neg_lambda)
    print(f"[train] neg_lambda={neg_lambda} n_pos={n_pos} n_neg={n_neg} -> "
          f"neg_weight={neg_weight:.4f} (negative unlikelihood term scale)")

    # --- LoRA config (attention projections only; NOT the MoE router/gate) ---
    from peft import LoraConfig
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules)

    # Router guard: refuse if any bare `gate` router module slipped into targets.
    gate_modules = [n for n, _ in model.named_modules() if "gate" in n]
    print(f"[train] modules containing 'gate' ({len(gate_modules)}): "
          f"{gate_modules[:20]}")
    assert not any(t == "gate" or t.endswith(".gate") for t in target_modules), \
        "refusing to LoRA the MoE router `gate` — unstable on this MoE"

    # --- trainer: subclass SFTTrainer, override compute_loss -----------------
    from trl import SFTConfig, SFTTrainer

    class _PrefCollator:
        """Wrap the PROVEN TRL collator. TRL builds input_ids / attention_mask /
        labels (assistant tokens kept, everything else -100 via assistant_masks);
        we strip the pass/fail `label` off each row BEFORE that collator runs (so
        it only sees tensor keys it knows) and re-attach a per-row `pref_is_pos`
        flag for compute_loss."""

        def __init__(self, base, label_key="label"):
            self.base = base
            self.label_key = label_key

        def __call__(self, examples):
            is_pos, cleaned = [], []
            for ex in examples:
                ex = dict(ex)
                lbl = ex.pop(self.label_key, None)
                # metadata that may ride along from export_preferences.
                for k in ("task", "trial", "run_id", "_ntok"):
                    ex.pop(k, None)
                if lbl not in ("pass", "fail"):
                    raise ValueError(
                        f"preference row has invalid {self.label_key!r}={lbl!r}; "
                        "expected 'pass' or 'fail'")
                is_pos.append(1.0 if lbl == "pass" else 0.0)
                cleaned.append(ex)
            batch = self.base(cleaned)
            batch["pref_is_pos"] = torch.tensor(is_pos, dtype=torch.float32)
            return batch

    class PrefSFTTrainer(SFTTrainer):
        def __init__(self, *args, neg_weight=0.0, neg_p_clamp=1e-4,
                     label_key="label", **kwargs):
            super().__init__(*args, **kwargs)
            self.neg_weight = neg_weight
            self.neg_p_clamp = neg_p_clamp
            # Wrap the collator TRL just built (self.data_collator is the proven
            # DataCollatorForLanguageModeling that applies the assistant mask).
            self.data_collator = _PrefCollator(self.data_collator,
                                               label_key=label_key)

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            # num_items_in_batch (global token normaliser) is intentionally
            # ignored: our objective is normalised per-row (mean over each row's
            # assistant tokens) + class-balanced via neg_weight, so a global token
            # count would double-normalise.
            pref_is_pos = inputs.pop("pref_is_pos")
            labels = inputs.pop("labels")
            outputs = model(**inputs)     # no labels -> model computes no loss
            loss = pref_loss_from_logits(outputs.logits, labels, pref_is_pos,
                                         self.neg_weight, self.neg_p_clamp)
            return (loss, outputs) if return_outputs else loss

    sft_config = SFTConfig(
        assistant_only_loss=True,          # reads the {% generation %} markers
        gradient_checkpointing=True,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        max_length=max_length,
        learning_rate=lr,                  # 5e-5: HALF of P4 (we also push away)
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        logging_steps=1,
        save_strategy="no",                # single explicit save at the end
        report_to="none",
        output_dir=out_dir,
        # KEEP the `label` column so _PrefCollator can read it — SFTTrainer's
        # signature columns are a fixed list that would otherwise strip it.
        remove_unused_columns=False,
    )

    if using_unsloth:
        from unsloth import FastLanguageModel as _FLM
        model = _FLM.get_peft_model(
            model, r=32, lora_alpha=64, lora_dropout=0, bias="none",
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth")
        trainer = PrefSFTTrainer(
            model=model, train_dataset=ds, args=sft_config,
            processing_class=tokenizer, neg_weight=neg_weight, label_key="label")
    else:
        trainer = PrefSFTTrainer(
            model=model, train_dataset=ds, args=sft_config,
            peft_config=lora_config, processing_class=tokenizer,
            neg_weight=neg_weight, label_key="label")

    train_out = trainer.train()
    final_loss = getattr(train_out, "training_loss", None)

    # --- persist adapter + tokenizer + patched template + meta ---------------
    trainer.save_model(out_dir)               # adapter weights
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "chat_template.jinja"), "w") as f:
        f.write(template)                     # the EXACT patched template used
    meta = {
        "base": MODEL,
        "method": ("weighted-CE unlikelihood on SFTTrainer(assistant_only_loss); "
                   "pos=token-CE, neg=-log(1-p) unlikelihood scaled by "
                   "neg_lambda*(n_pos/n_neg)"),
        "r": 32, "lora_alpha": 64, "target_modules": target_modules,
        "lr": lr, "epochs": epochs, "neg_lambda": neg_lambda,
        "neg_weight": neg_weight, "max_length": max_length,
        "n_pos": n_pos, "n_neg": n_neg,
        "n_filtered_pos": n_filt_pos, "n_filtered_neg": n_filt_neg,
        "using_unsloth": using_unsloth, "final_loss": final_loss,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    adapters.commit()
    print(f"[train] done: {out_dir} — {meta}")
    return meta


# --- local entrypoint ------------------------------------------------------

@app.local_entrypoint()
def main(step="train", epochs=1, lr=5e-5, max_length=32768, neg_lambda=0.2,
         run_id=None):
    """Dispatch a P7 step:

      modal run infra/train_pref.py::preflight_pref          # CPU gate (run FIRST)
      modal run infra/train_pref.py::main --step train [--epochs N --lr ... \\
                                                        --neg-lambda 0.2]

    The bf16 weights are shared with P4 — populate them once via
    `modal run infra/train_lora.py::main --step download`. `train` relies on the
    preference JSONL baked into train_pref_image at /data/pref.jsonl (PREF_PATH at
    build time).
    """
    if step == "train":
        print(train_pref.remote(pref_path_in_image="/data/pref.jsonl",
                                 epochs=epochs, lr=lr, max_length=max_length,
                                 neg_lambda=neg_lambda, run_id=run_id))
    elif step == "preflight":
        print("run the CPU gate directly: "
              "modal run infra/train_pref.py::preflight_pref")
    elif step == "download":
        print("weights are shared with P4 — run: "
              "modal run infra/train_lora.py::main --step download")
    else:
        raise SystemExit(f"unknown step {step!r}: use preflight|download|train")
