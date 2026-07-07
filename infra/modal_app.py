"""Modal app for the Ornith → Claude Code adaptation project.

Three entrypoints:
  modal deploy infra/modal_app.py                    # serve vLLM endpoint
  modal run infra/modal_app.py::smoke                # Phase 0 gate
  modal run infra/modal_app.py::run_sweep --variant v001-baseline --trials 5

Skeleton status: structure and invariants are real; bodies marked TODO are
Phase 0 work. Everything cost- or verdict-bearing is code here, never model
procedure — the runner writes runs/, summary.json, and cost-ledger.csv itself.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

MODEL = "deepreinforce-ai/Ornith-1.0-35B-FP8"
GPU = "H100"
MINUTES = 60

app = modal.App("ornith-harness")

# Weights cached once; ~35GB FP8 keeps cold starts ~1-2 min.
weights = modal.Volume.from_name("ornith-weights", create_if_missing=True)
runs_vol = modal.Volume.from_name("ornith-runs", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.19.1", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # TODO(phase0): bake the patched chat_template.jinja (community PR removes
    # the raise_exception asserts) into the image and pass --chat-template.
)


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={"/weights": weights},
    scaledown_window=120,  # warm containers bill; keep this short
    timeout=60 * MINUTES,
)
@modal.web_server(port=8000, startup_timeout=10 * MINUTES)
def serve():
    """OpenAI-compatible vLLM endpoint. Claude Code connects via an
    Anthropic-compat proxy (claude-code-router / LiteLLM) pointed here."""
    cmd = [
        "vllm", "serve", MODEL,
        "--download-dir", "/weights",
        "--served-model-name", "ornith-35b",
        "--max-model-len", "131072",  # raise after Phase 0 if KV budget allows
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--reasoning-parser", "qwen3",
        "--enable-prefix-caching",
        "--trust-remote-code",
        # TODO(phase0): --chat-template /weights/chat_template_patched.jinja
    ]
    subprocess.Popen(cmd)


@app.local_entrypoint()
def smoke():
    """Phase 0 gate (G1): schema-clean tool calls, no <think> leakage,
    20/20 trivial tasks through the real proxy path. Exits nonzero on any
    failure so it can gate CI.

    TODO(phase0): implement — call the deployed endpoint through the proxy,
    assert on: (1) tool_calls parse against Anthropic schema, (2) content
    contains no '<think>', (3) trivial task pass count == 20.
    """
    raise SystemExit("smoke: not implemented (Phase 0)")


@app.function(image=vllm_image, volumes={"/runs": runs_vol}, timeout=120 * MINUTES)
def run_trial(task_spec: dict, variant_config_tar: bytes, trial_idx: int) -> dict:
    """One task x one trial, inside the task's pinned container.

    Invariants this function owns (not the model, not the skill):
      - materializes variant claude-config/ as .claude/ INSIDE the task
        workspace only, and writes task_spec['verify'] to VERIFY.txt at the
        workspace root (the worker CLAUDE.md tells the model to use it);
      - runs the worker (Claude Code CLI against the ornith endpoint) with
        the task description;
      - executes task_spec['verify']; the exit code is the verdict;
      - records tokens, gpu_seconds, wall_clock, transcript;
      - an execution error is verdict='invalid', never 'fail'.

    TODO(phase0/1): implement against the task container runtime.
    """
    started = time.time()
    return {
        "task": task_spec.get("id"),
        "trial": trial_idx,
        "verdict": "invalid",
        "error": "run_trial not implemented",
        "wall_clock_s": round(time.time() - started, 2),
        "tokens_in": 0,
        "tokens_out": 0,
        "gpu_seconds": 0.0,
    }


@app.local_entrypoint()
def run_sweep(variant: str, trials: int = 5, split: str = "dev"):
    """Fan a variant out across the task set; write runs/<run_id>/ and append
    the cost ledger. Holdout runs require the operator's .holdout-unlocked
    flow upstream — this entrypoint refuses split='holdout' unless the flag
    file exists, as defense in depth."""
    root = Path(__file__).resolve().parent.parent
    if split == "holdout" and not (root / ".holdout-unlocked").exists():
        raise SystemExit("holdout sealed: operator must create .holdout-unlocked")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{variant}"
    task_dirs = sorted((root / "tasks" / split).glob("*/task.yaml"))
    if not task_dirs:
        raise SystemExit(f"no tasks in tasks/{split}/")

    # TODO(phase1): load specs, tar the variant's claude-config/, then:
    #   results = list(run_trial.starmap(work_items))
    # then write summary.json (paired stats vs parent, provenance slices)
    # and append findings/cost-ledger.csv: run_id,timestamp,gpu_seconds,usd.
    raise SystemExit(f"run_sweep skeleton: found {len(task_dirs)} tasks; runner body is Phase 1 work (run {run_id})")
