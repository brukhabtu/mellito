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
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # This slim image has no CUDA toolkit (no nvcc). FlashInfer's
            # top-k/top-p sampler JIT-compiles at boot and hard-fails without
            # nvcc (RuntimeError in flashinfer.jit). Force vLLM's native Torch
            # sampler — output-identical at temperature 0 (greedy) and needs no
            # JIT. (Full-attn=FlashAttention, MoE=CUTLASS, GDN-prefill=triton
            # are all already JIT-free.)
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={"/weights": weights},
    scaledown_window=120,  # warm containers bill; keep this short
    timeout=60 * MINUTES,
)
# First boot downloads ~35GB of FP8 shards into the volume before vLLM can
# bind :8000; subsequent boots hit the cached volume (~1-2 min). Give the
# startup probe room for the cold pull.
@modal.web_server(port=8000, startup_timeout=30 * MINUTES)
def serve():
    """OpenAI-compatible vLLM endpoint for Ornith-1.0-35B-FP8.

    The model is a Qwen3.5-MoE (35B-A3B: 3B active / 256 experts) hybrid
    linear+full-attention VL model; vLLM >=0.17 registers
    Qwen3_5MoeForConditionalGeneration. We serve it text-only. Claude Code
    connects via an Anthropic-compat proxy (claude-code-router / LiteLLM)
    pointed here; the proxy is a field-rename over this OpenAI schema.

    Chat template: the model ships chat_template.jinja (the qwen3_xml
    <tool_call><function=..><parameter=..> format). Its raise_exception sites
    are input-validation guards (images-in-system, no-messages, bad-role) that
    do not fire on well-formed text tool-use, and vLLM provides raise_exception
    in its jinja sandbox — so we load the bundled template as-is (no override).
    """
    cmd = [
        "vllm", "serve", MODEL,
        "--download-dir", "/weights",
        "--served-model-name", "ornith-35b",
        # Conservative for the G1 first boot on a single H100; raise once KV
        # headroom is measured (model supports up to 262144).
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--reasoning-parser", "qwen3",
        "--enable-prefix-caching",
        "--trust-remote-code",
        # The gated-delta-net linear-attention layers default to FlashInfer's
        # GDN prefill kernel, which JIT-compiles via nvcc — absent from this
        # slim pip image (crash loop). Triton JIT needs no nvcc.
        "--gdn-prefill-backend", "triton",
        # G1 boot: skip torch.compile + cudagraph capture for a fast, robust
        # cold start. Output is identical to compiled mode; revisit for
        # throughput in P2 (bake nvcc / re-enable compile if worth it).
        "--enforce-eager",
    ]
    subprocess.Popen(cmd)


# --- G1 smoke suite -------------------------------------------------------

# 20 trivial, deterministic prompts. Each expected answer is a distinctive
# token so a lenient case-insensitive substring check can't match spuriously.
# Trials run at temperature 0 with reasoning ON, so a pass proves both that the
# model solves the trivial and that the reasoning parser stripped <think>.
TRIVIALS = [
    ("What is 7 times 6? Reply with only the number.", "42"),
    ("What is 100 minus 37? Reply with only the number.", "63"),
    ("What is 12 plus 31? Reply with only the number.", "43"),
    ("What is 9 squared? Reply with only the number.", "81"),
    ("What is 144 divided by 12? Reply with only the number.", "12"),
    ("How many sides does a hexagon have? Reply with only the number.", "6"),
    ("What is the capital of Japan? Reply with only the city name.", "tokyo"),
    ("What is the capital of France? Reply with only the city name.", "paris"),
    ("What color do you get mixing blue and yellow? One word.", "green"),
    ("Reply with exactly this word and nothing else: pomegranate", "pomegranate"),
    ("Reply with exactly this word and nothing else: helicopter", "helicopter"),
    ("What is the chemical symbol for gold? Reply with only the symbol.", "au"),
    ("What planet is known as the Red Planet? One word.", "mars"),
    ("What is the opposite of 'up'? One word.", "down"),
    ("How many days are in a week? Reply with only the number.", "7"),
    ("What is the first month of the year? One word.", "january"),
    ("What gas do plants absorb that humans exhale? Two words.", "carbon dioxide"),
    ("Spell the word 'cat' backwards. Reply with only that.", "tac"),
    ("What is 2 to the power of 5? Reply with only the number.", "32"),
    ("What is the largest ocean on Earth? One word (the name only).", "pacific"),
]

# One forced tool call — proves the qwen3_xml parser emits schema-clean
# OpenAI tool_calls that map cleanly onto an Anthropic tool_use block.
ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return their sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "first addend"},
                "b": {"type": "integer", "description": "second addend"},
            },
            "required": ["a", "b"],
        },
    },
}


def _serve_url() -> str:
    import modal as _modal
    fn = _modal.Function.from_name("ornith-harness", "serve")
    for attr in ("get_web_url", "web_url"):
        v = getattr(fn, attr, None)
        url = v() if callable(v) else v
        if url:
            return url.rstrip("/")
    raise SystemExit("smoke: could not resolve serve() web URL — is it deployed?")


def _wait_healthy(url: str, timeout_s: int = 30 * 60) -> None:
    import time
    import urllib.error
    import urllib.request
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError) as e:  # cold start / not up yet
            last = str(e)
        time.sleep(10)
    raise SystemExit(f"smoke: endpoint not healthy after {timeout_s}s ({last})")


def _chat(url: str, messages, tools=None, tool_choice=None, max_tokens=1024) -> dict:
    import json
    import urllib.request
    body = {
        "model": "ornith-35b",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


@app.local_entrypoint()
def smoke():
    """Phase 0 gate (G1). Exits nonzero on any failure so it can gate CI.

    Asserts, against the deployed endpoint (the same OpenAI schema the
    Anthropic proxy renames field-for-field):
      (1) tool_calls parse into a valid Anthropic tool_use block,
      (2) no response content contains '<think>' / '</think>',
      (3) all 20 trivial tasks are answered correctly (20/20).
    """
    import json

    url = _serve_url()
    print(f"smoke: endpoint {url}")
    _wait_healthy(url)
    print("smoke: endpoint healthy")

    failures = []
    leaks = 0

    # (1) + (3): trivials, checking correctness and <think> leakage together.
    passed = 0
    for i, (prompt, expect) in enumerate(TRIVIALS, 1):
        try:
            resp = _chat(url, [{"role": "user", "content": prompt}])
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
        except Exception as e:  # execution error is a smoke failure, not a pass
            failures.append(f"trivial {i}: request error: {e}")
            continue
        if "<think>" in content or "</think>" in content:
            leaks += 1
            failures.append(f"trivial {i}: <think> leaked into content")
        if expect.lower() in content.lower():
            passed += 1
        else:
            failures.append(f"trivial {i}: expected {expect!r} not in {content[:120]!r}")
    print(f"smoke: trivials {passed}/{len(TRIVIALS)} correct; think-leaks {leaks}")

    # (2)/(1): forced tool call → schema-clean → maps to Anthropic tool_use.
    tool_ok = False
    try:
        resp = _chat(
            url,
            [{"role": "user", "content": "Use the add tool to compute 12345 + 67890."}],
            tools=[ADD_TOOL],
            tool_choice="auto",
        )
        msg = resp["choices"][0]["message"]
        if "<think>" in (msg.get("content") or ""):
            failures.append("tool call: <think> leaked into content")
        tcs = msg.get("tool_calls") or []
        if not tcs:
            failures.append("tool call: no tool_calls returned")
        else:
            tc = tcs[0]
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])  # must be valid JSON
            # Map OpenAI tool_call -> Anthropic tool_use and validate shape.
            anthropic_block = {"type": "tool_use", "name": name, "input": args}
            if (
                name == "add"
                and isinstance(args, dict)
                and set(args) >= {"a", "b"}
                and isinstance(anthropic_block["input"], dict)
            ):
                tool_ok = True
            else:
                failures.append(f"tool call: bad schema {anthropic_block!r}")
    except Exception as e:
        failures.append(f"tool call: request error: {e}")
    print(f"smoke: tool-call schema {'OK' if tool_ok else 'FAILED'}")

    ok = passed == len(TRIVIALS) and leaks == 0 and tool_ok
    if not ok:
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(
            f"smoke: FAILED ({passed}/{len(TRIVIALS)} trivials, "
            f"{leaks} think-leaks, tool_ok={tool_ok})"
        )
    print("smoke: PASS (20/20 trivials, 0 think-leaks, schema-clean tool call)")


@app.function(image=vllm_image, volumes={"/runs": runs_vol}, timeout=120 * MINUTES)
def run_trial(task_spec: dict, variant_config_tar: bytes, trial_idx: int) -> dict:
    """One task x one trial, inside the task's pinned container.

    Invariants this function owns (not the model, not the skill):
      - materializes variant claude-config/ as .claude/ INSIDE the task
        workspace only, and writes task_spec['verify'] to VERIFY.txt at the
        workspace root (the worker CLAUDE.md tells the model to use it);
      - runs the worker (Claude Code CLI against the ornith endpoint) with
        the task description;
      - VERDICT (hidden-tests contract, see tasks/schema.md): after the worker
        finishes, reset to base + reapply the worker's diff, then
        `git apply` task_spec['hidden_tests'] (the tests.patch beside task.yaml)
        and run task_spec['verify']; the exit code is the verdict. The tests are
        injected ONLY here — never in the worker's workspace — so the model
        cannot read or edit them (corpus-curator item 6: no oracle leakage).
        Verified locally that stored tests flip fail->pass for a gold fix.
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
