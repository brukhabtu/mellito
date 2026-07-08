"""Modal app for the P4 Ornith bf16 LoRA fit — separate app so it never
disturbs the deployed `ornith-harness` serving stack.

Three steps, driven by `modal run infra/train_lora.py::main --step <step>`:

  preflight  — download only the tokenizer + chat template (a few MB, no GPU,
               no weights), patch the template for assistant_only_loss if the
               stock template lacks generation markers, render a couple of real
               SFT examples through it, and EYEBALL that the assistant-token mask
               covers the response (thinking in <think>, tool calls in
               <tool_call>/<function>) and is non-empty. This is the empirical
               gate that the template patch is correct BEFORE any GPU time.
  download   — snapshot the ~70GB bf16 weights into a persistent Volume once.
  train      — the actual LoRA fit on one H100.

WHY bf16 (not the FP8 serving checkpoint): LoRA fine-tuning needs a
full-precision base to add trainable adapters onto; FP8 is a serving-time
quantization. Serving stays on FP8 (modal_app.py); training pulls the bf16 repo
into its own Volume. The two never share weights.

The train==inference invariant lives in chat_template_adapt.py: we serialize the
Anthropic-shaped SFT turns through the MODEL'S OWN qwen3_xml template, patched
only to add the loss-mask markers. See that module's docstring.

Structural note: this file is built to import cleanly and be structurally
correct; the GPU/Modal bodies are NOT exercised in the harness (no GPU, no Modal
runtime here). The proof they run is the preflight step on Modal.
"""

import os

import modal

# Default location of the SFT JSONL produced by export_trajectories.py. It is
# gitignored/ephemeral (the scratchpad), so it is added to the train image at
# build time from SFT_PATH; override with `SFT_PATH=/abs/path modal run ...`.
_DEFAULT_SFT_PATH = (
    "/tmp/claude-0/-home-user-mellito/"
    "5ffb3023-5238-5cf4-a306-ed31d13a97c9/scratchpad/ornith_sft.jsonl"
)
SFT_PATH = os.environ.get("SFT_PATH", _DEFAULT_SFT_PATH)

MODEL = "deepreinforce-ai/Ornith-1.0-35B"  # bf16 repo (NOT the -FP8 serving one)
GPU = "H100"
MINUTES = 60

app = modal.App("ornith-lora")

# bf16 base weights (~70GB) cached once; adapters are tiny LoRA deltas + meta.
weights_bf16 = modal.Volume.from_name("ornith-weights-bf16", create_if_missing=True)
adapters = modal.Volume.from_name("ornith-adapters", create_if_missing=True)

# HF token only needed if the repo turns out to be gated. Modal resolves
# Secret.from_name lazily at RUN time (not construction), so a try/except here
# can't shield a missing secret — it would fail the run. Per the plan the bf16
# repo pulls tokenless (its FP8 sibling does in serve()); only if a pull 401s do
# we create the secret. So attach nothing by default; to enable, create the
# secret (`modal secret create huggingface-token HF_TOKEN=hf_...`) and set this
# to [modal.Secret.from_name("huggingface-token")].
_HF_SECRETS: list = []


# --- images ----------------------------------------------------------------

# Slim CPU image just for pulling weights: hf_transfer for fast sharded download.
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Preflight needs a tokenizer + jinja + our pure helpers, but NO torch/GPU.
preflight_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers", "jinja2", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("chat_template_adapt", "export_trajectories")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "ornith_chat_template.jinja"),
        "/ornith_chat_template.jinja")
)

# Training image: TRL+PEFT (the arch-agnostic path). Research indicates Unsloth
# does not register this Qwen3.5-MoE hybrid Mamba/GDN checkpoint, so we do NOT
# bake the heavy/fragile unsloth wheel into the image — train_lora attempts
# `import unsloth` at runtime and falls straight to TRL+PEFT if it's absent or
# can't load the arch. The SFT JSONL is baked in at /data/sft.jsonl.
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "jinja2",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("chat_template_adapt", "export_trajectories")
    .add_local_file(SFT_PATH, "/data/sft.jsonl")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "ornith_chat_template.jinja"),
        "/ornith_chat_template.jinja")
)


# --- step 1: download bf16 weights ----------------------------------------

@app.function(image=download_image, volumes={"/weights_bf16": weights_bf16},
              timeout=90 * MINUTES, secrets=_HF_SECRETS)
def download_bf16_weights():
    """Snapshot the ~70GB bf16 Ornith repo into the weights_bf16 Volume once.

    Sentinel-skips if a config.json + any *.safetensors already exist (a warm
    Volume). Downloads tokenlessly first (the repo is expected public); on a
    401/gated error, retries with HF_TOKEN if a huggingface-token secret is set.
    """
    import glob
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError

    dst = "/weights_bf16"
    have_config = os.path.exists(os.path.join(dst, "config.json"))
    have_shards = bool(glob.glob(os.path.join(dst, "*.safetensors")))
    if have_config and have_shards:
        print(f"[download] weights already present in {dst} — skipping")
        return {"skipped": True, "dir": dst}

    # Skip .pt duplicates, the original/ fork, and any gguf — we only need the
    # safetensors bf16 shards + config/tokenizer/template.
    ignore = ["*.pt", "original/*", "*.gguf"]
    try:
        snapshot_download(MODEL, local_dir=dst, ignore_patterns=ignore)
    except (GatedRepoError, Exception) as e:  # 401 / gated -> retry with token
        token = os.environ.get("HF_TOKEN")
        is_auth = isinstance(e, GatedRepoError) or "401" in str(e)
        if not (is_auth and token):
            raise
        print("[download] tokenless pull failed on auth; retrying with HF_TOKEN")
        snapshot_download(MODEL, local_dir=dst, ignore_patterns=ignore, token=token)

    weights_bf16.commit()
    print(f"[download] committed weights to {dst}")
    return {"skipped": False, "dir": dst}


# --- step 2: preflight the chat template ----------------------------------

@app.function(image=preflight_image, timeout=15 * MINUTES, secrets=_HF_SECRETS)
def preflight_template(samples):
    """Verify the patched chat template produces a correct assistant loss mask.

    Pulls ONLY the tokenizer + template files (a few MB), patches the template if
    it lacks `{% generation %}` markers, and renders each sample dict through it
    with the real tokenizer. For each, asserts:
      - the assistant-masked span is non-empty,
      - thinking renders inside <think>...</think>,
      - tool calls render with <tool_call> / <function ...> markup.
    Prints a truncated render + the decoded masked span for a human eyeball, and
    returns a report dict. Raises (FAIL) on any failed assertion.
    """
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

    stock = tok.chat_template
    if stock is None:
        raise SystemExit("preflight FAIL: tokenizer has no chat_template")

    # Ornith's stock template uses `message.role == "assistant"` attribute-form
    # and emits the header+think+content in one expression, which the generic
    # cta.patch_template can't splice. We ship a hand-patched copy
    # (infra/ornith_chat_template.jinja) verified locally to render BYTE-IDENTICAL
    # to stock (the generation markers add no output) with an assistant-only mask
    # that covers <think>+text+tool_call and excludes tool_response/system/user.
    # The gate here re-asserts that byte-identity against the live model template,
    # so a silent upstream template change is caught before any GPU spend.
    if cta.has_generation_markers(stock):
        patched = stock
        print("[preflight] stock template already has generation markers")
    else:
        with open("/ornith_chat_template.jinja") as f:
            patched = f.read()
        # byte-identical-render safety check on the first sample
        m0 = cta.anthropic_to_template_messages(samples[0]["messages"])
        tok.chat_template = stock
        s_txt = tok.apply_chat_template(m0, tokenize=False)
        tok.chat_template = patched
        p_txt = tok.apply_chat_template(m0, tokenize=False)
        if s_txt != p_txt:
            raise SystemExit("preflight FAIL: committed patched template does not "
                             "render byte-identical to the model's stock template "
                             "— the upstream chat_template changed; re-patch "
                             "infra/ornith_chat_template.jinja")
        print("[preflight] committed patched template renders byte-identical to "
              "stock; generation markers added")

    reports = []
    all_ok = True
    for i, ex in enumerate(samples):
        rep = cta.render_and_verify(tok, patched, ex)
        text = rep["text"]
        # Decode the first masked span for the human eyeball.
        spans = rep["assistant_mask_spans"]
        span_txt = ""
        if spans:
            enc = tok.apply_chat_template(
                cta.anthropic_to_template_messages(ex["messages"]),
                tokenize=True, return_dict=True)
            ids = enc["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            s, e = spans[0]
            span_txt = tok.decode(ids[s:e])

        has_thinking = any(
            b.get("type") == "thinking"
            for m in ex["messages"] if isinstance(m.get("content"), list)
            for b in m["content"])
        has_tool = any(
            b.get("type") == "tool_use"
            for m in ex["messages"] if isinstance(m.get("content"), list)
            for b in m["content"])

        checks = {"mask_nonempty": rep["n_assistant_tokens"] > 0}
        if has_thinking:
            checks["thinking_in_think_tags"] = (
                "<think>" in text and "</think>" in text)
        if has_tool:
            checks["tool_call_markup"] = (
                "<tool_call>" in text or "<function" in text)

        ok = all(checks.values())
        all_ok = all_ok and ok
        print(f"\n[preflight] sample {i} ({ex.get('task')}): "
              f"{'PASS' if ok else 'FAIL'} checks={checks}")
        print(f"  n_tokens={rep['n_tokens']} "
              f"n_assistant_tokens={rep['n_assistant_tokens']} "
              f"spans={rep['assistant_mask_spans'][:4]}")
        print(f"  render[:600]: {text[:600]!r}")
        print(f"  masked-span[:400]: {span_txt[:400]!r}")
        reports.append({"task": ex.get("task"), "ok": ok, "checks": checks,
                        "n_assistant_tokens": rep["n_assistant_tokens"],
                        "n_tokens": rep["n_tokens"]})

    print(f"\n[preflight] {'PASS' if all_ok else 'FAIL'} "
          f"({sum(r['ok'] for r in reports)}/{len(reports)} samples ok)")
    if not all_ok:
        raise SystemExit("preflight FAIL: see per-sample checks above")
    return {"had_markers": had_markers, "n_samples": len(samples),
            "reports": reports}


# --- step 3: the LoRA fit --------------------------------------------------

@app.function(image=train_image,
              volumes={"/weights_bf16": weights_bf16, "/adapters": adapters},
              gpu=GPU, timeout=180 * MINUTES, secrets=_HF_SECRETS)
def train_lora(sft_path_in_image="/data/sft.jsonl", epochs=2, lr=1e-4,
               max_length=32768, run_id=None):
    """Fit a bf16 LoRA on the passing-trajectory SFT set on one H100.

    Unsloth-primary / TRL+PEFT-fallback (chosen at runtime via try/except so the
    fallback needs no image rebuild). Serializes turns through the model's own
    qwen3_xml template (patched for assistant_only_loss). Length-FILTERS overlong
    examples rather than truncating (a truncated target would drop the tail of a
    trajectory mid-tool-call and teach a broken shape). Saves the adapter,
    tokenizer, patched template, and a run_meta.json into the adapters Volume.
    """
    import glob
    import json
    import time

    import torch
    from datasets import load_dataset

    import chat_template_adapt as cta

    run_id = run_id or f"lora-{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir = f"/adapters/{run_id}"
    weights_dir = "/weights_bf16"
    if not glob.glob(os.path.join(weights_dir, "*.safetensors")):
        raise SystemExit(
            f"no bf16 weights in {weights_dir} — run `--step download` first")

    # --- load base model + tokenizer: Unsloth first, eager HF fallback -------
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
        model = AutoModelForCausalLM.from_pretrained(
            weights_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="eager")

    # --- install the patched chat template (assistant_only_loss markers) -----
    # Use the committed, preflight-verified patched template (byte-identical to
    # stock, adds only the generation markers). preflight_template is the gate
    # that proves this against the live model template.
    stock = tokenizer.chat_template
    if stock is None:
        raise SystemExit("train FAIL: tokenizer has no chat_template")
    if cta.has_generation_markers(stock):
        template = stock
    else:
        with open("/ornith_chat_template.jinja") as f:
            template = f.read()
        print("[train] installed committed patched chat_template "
              "(generation markers for assistant_only_loss)")
    tokenizer.chat_template = template

    # --- load + shape the SFT set -------------------------------------------
    # datasets reads the JSONL; .map renames each example's Anthropic block-list
    # messages into the stock-template dicts our template consumes.
    ds = load_dataset("json", data_files=sft_path_in_image, split="train")

    def _shape(ex):
        return {"messages": cta.anthropic_to_template_messages(ex["messages"])}

    ds = ds.map(_shape, remove_columns=[c for c in ds.column_names
                                        if c != "messages"])

    # Length-FILTER (never truncate): drop examples whose tokenized length under
    # this template exceeds max_length. A truncated trajectory would teach a
    # broken tail; dropping keeps every kept target complete.
    def _fits(ex):
        ids = tokenizer.apply_chat_template(ex["messages"], tokenize=True)
        n = len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
        return n <= max_length

    n_before = len(ds)
    ds = ds.filter(_fits)
    n_filtered = n_before - len(ds)
    print(f"[train] length-filter: kept {len(ds)}/{n_before} "
          f"(dropped {n_filtered} over {max_length} tokens)")
    if len(ds) == 0:
        raise SystemExit("train FAIL: all examples filtered out by max_length")

    # --- LoRA config ---------------------------------------------------------
    # Attention projections only. Deliberately NOT the MoE router/gate or expert
    # MLPs: adapting the router is unstable and expert-MLP LoRA on a 256-expert
    # MoE is both huge and risky. r=32/alpha=64 is a conservative capacity.
    from peft import LoraConfig
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules)

    # Router guard: assert no bare `gate` router module slipped into the targets.
    # Printing the gate-bearing module names also documents the arch for the log.
    gate_modules = [n for n, _ in model.named_modules() if "gate" in n]
    print(f"[train] modules containing 'gate' ({len(gate_modules)}): "
          f"{gate_modules[:20]}")
    assert not any(t == "gate" or t.endswith(".gate") for t in target_modules), \
        "refusing to LoRA the MoE router `gate` — unstable on this MoE"

    # --- trainer -------------------------------------------------------------
    from trl import SFTConfig, SFTTrainer
    sft_config = SFTConfig(
        assistant_only_loss=True,       # reads the {% generation %} markers
        gradient_checkpointing=True,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        max_length=max_length,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        output_dir=out_dir,
    )

    # Unsloth wants its own PEFT-attach entrypoint; TRL takes peft_config direct.
    if using_unsloth:
        from unsloth import FastLanguageModel as _FLM
        model = _FLM.get_peft_model(
            model, r=32, lora_alpha=64, lora_dropout=0, bias="none",
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth")
        trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_config,
                             processing_class=tokenizer)
    else:
        trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_config,
                             peft_config=lora_config,
                             processing_class=tokenizer)

    train_out = trainer.train()
    final_loss = getattr(train_out, "training_loss", None)

    # --- persist adapter + tokenizer + patched template + meta ---------------
    trainer.save_model(out_dir)               # adapter weights
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "chat_template.jinja"), "w") as f:
        f.write(template)                     # the EXACT patched template used
    meta = {
        "base": MODEL, "r": 32, "lora_alpha": 64,
        "target_modules": target_modules, "epochs": epochs, "lr": lr,
        "using_unsloth": using_unsloth, "max_length": max_length,
        "n_examples": len(ds), "n_filtered": n_filtered,
        "final_loss": final_loss,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    adapters.commit()
    print(f"[train] done: {out_dir} — {meta}")
    return meta


# --- local entrypoint ------------------------------------------------------

def _read_sample_lines(n=3):
    """Read up to n examples from the local SFT JSONL for preflight to render.

    preflight_template receives these as an argument (rather than reading a file
    inside the container) so it needs no weights volume and no baked-in data —
    just the tokenizer download + the pure helpers.
    """
    import json
    samples = []
    try:
        with open(SFT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
                if len(samples) >= n:
                    break
    except FileNotFoundError:
        raise SystemExit(
            f"SFT jsonl not found at {SFT_PATH} — set SFT_PATH to the export "
            "produced by export_trajectories.py")
    if not samples:
        raise SystemExit(f"no examples read from {SFT_PATH}")
    return samples


@app.local_entrypoint()
def main(step="preflight", epochs=2, lr=1e-4, max_length=32768, run_id=None):
    """Dispatch a P4 step:

      modal run infra/train_lora.py::main --step preflight
      modal run infra/train_lora.py::main --step download
      modal run infra/train_lora.py::main --step train [--epochs N --lr ...]

    preflight passes local SFT sample lines to the remote fn; train relies on the
    SFT jsonl baked into train_image at /data/sft.jsonl (SFT_PATH at build time).
    """
    if step == "download":
        print(download_bf16_weights.remote())
    elif step == "preflight":
        samples = _read_sample_lines(3)
        print(preflight_template.remote(samples))
    elif step == "train":
        print(train_lora.remote(sft_path_in_image="/data/sft.jsonl",
                                 epochs=epochs, lr=lr, max_length=max_length,
                                 run_id=run_id))
    else:
        raise SystemExit(f"unknown step {step!r}: use preflight|download|train")
