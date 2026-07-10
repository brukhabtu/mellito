"""Modal app for the Ornith → Claude Code adaptation project.

Three entrypoints:
  modal deploy infra/modal_app.py                    # serve vLLM endpoint
  modal run infra/modal_app.py::smoke                # Phase 0 gate
  modal run infra/modal_app.py::run_sweep --variant v001-baseline --trials 5

Skeleton status: structure and invariants are real; bodies marked TODO are
Phase 0 work. Everything cost- or verdict-bearing is code here, never model
procedure — the runner writes runs/, summary.json, and cost-ledger.csv itself.
"""

import io
import json
import os
import re
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal

MODEL = "deepreinforce-ai/Ornith-1.0-35B-FP8"
GPU = "H100"
MINUTES = 60

# Endpoint auth keys live in a Modal secret (never in code): VLLM_API_KEY gates
# vLLM's /v1/* (serve), and doubles as LiteLLM's upstream api_key; the proxy's
# own /v1/* is gated by PROXY_MASTER_KEY. Both /health probes stay public.
ENDPOINT_KEYS = modal.Secret.from_name("ornith-endpoint-keys")

app = modal.App("ornith-harness")

# Weights cached once; ~35GB FP8 keeps cold starts ~1-2 min.
weights = modal.Volume.from_name("ornith-weights", create_if_missing=True)
runs_vol = modal.Volume.from_name("ornith-runs", create_if_missing=True)
# P4 LoRA adapters (written by infra/train_lora.py). Mounted read-only into
# serve() so vLLM can hot-load the rank-32 adapter alongside the FP8 base.
adapters = modal.Volume.from_name("ornith-adapters", create_if_missing=True)

# The P4 adapter served next to the base. bf16 LoRA (r=32, attention-only)
# trained on Ornith's own verifier-passing trajectories; see FINDINGS
# 2026-07-08 P4 LoRA. Applied over the FP8 base — attention projection shapes
# are identical across the FP8/bf16 siblings, so the bf16 delta drops in.
LORA_ADAPTER = "pref-20260710T120439"

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
    volumes={"/weights": weights, "/adapters": adapters},
    scaledown_window=120,  # warm containers bill; keep this short
    timeout=60 * MINUTES,
    secrets=[ENDPOINT_KEYS],
)
# Let ONE container serve many requests at once. Without this, Modal routes a
# single request per container to the web server, so vLLM's continuous batcher
# never sees a queue (measured: 8 concurrent clients -> "Running: 1 reqs" in the
# engine, ~13 tok/s aggregate = single-stream). With concurrent inputs, vLLM
# batches across the fleet of trials, multiplying aggregate throughput on the
# one H100 — the lever that makes a G3 sweep affordable.
@modal.concurrent(max_inputs=64)
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
        # 128k window. Claude Code's system prompt + tool schemas are ~11k
        # tokens before any task context, and it sizes its output request to
        # fill the window — 32k overflowed on the very first agentic call
        # (ContextWindowExceededError). Ornith is a Mamba/GDN hybrid, so only
        # the few full-attention layers grow KV with length; a large window is
        # cheap here (model supports up to 262144). Drop if boot KV OOMs.
        "--max-model-len", "131072",
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
        # P2 throughput: cudagraph + torch.compile (inductor->triton, no nvcc)
        # remove the per-decode-step Python/launch overhead that pinned eager
        # mode to ~15 tok/s single-stream. Cap capture sizes so cold-boot graph
        # capture stays bounded (the reason eager was used at G1). Sampler and
        # GDN-prefill stay on their nvcc-free backends (set above).
        "--compilation-config",
        '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}',
        # vLLM's --api-key middleware gates /v1/* (Bearer). /health stays public,
        # so _wait_healthy needs no key. Key comes from the Modal secret.
        "--api-key", os.environ["VLLM_API_KEY"],
        # P4: hot-load the rank-32 LoRA as a SECOND served model ("ornith-lora")
        # alongside the base ("ornith-35b"). Both are reachable on the same
        # endpoint, so the re-gate can run the tuned model paired against the
        # base with no second deployment. --max-lora-rank must be >= the
        # adapter's r (32). If vLLM rejects LoRA on this hybrid MoE/FP8 arch at
        # boot, the fallback is merge_and_unload -> bf16 -> FP8 requantize (see
        # plan Step D); the base model stays served either way.
        "--enable-lora",
        "--lora-modules", f"ornith-lora=/adapters/{LORA_ADAPTER}",
        "--max-lora-rank", "32",
    ]
    subprocess.Popen(cmd)


# --- Anthropic-compat proxy ----------------------------------------------

# Claude Code speaks the Anthropic /v1/messages schema; vLLM (serve()) speaks
# the OpenAI schema. LiteLLM's proxy is the field-rename bridge between them.
# Pinned to the newest stable release at build time (1.91.0; checked against
# PyPI). Bump only in a `harness:` commit.
proxy_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "litellm[proxy]==1.91.0"
)


@app.function(image=proxy_image, scaledown_window=300, timeout=60 * MINUTES,
              secrets=[ENDPOINT_KEYS])
# One proxy container fans many concurrent worker requests onto the shared vLLM
# fleet; match serve()'s concurrency so the proxy is never the bottleneck.
@modal.concurrent(max_inputs=64)
@modal.web_server(port=4000, startup_timeout=5 * MINUTES)
def proxy():
    """LiteLLM proxy exposing an Anthropic-format endpoint backed by vLLM.

    Resolves serve()'s deployed URL at boot and writes a one-model config
    routing Anthropic requests for `ornith-35b` to `openai/ornith-35b` at
    serve()/v1. drop_params tolerates Anthropic-only fields vLLM doesn't accept.
    The upstream api_key is VLLM_API_KEY (serve's Bearer gate); the proxy's own
    /v1/* is gated by PROXY_MASTER_KEY. Both come from the Modal secret.
    """
    serve_url = _serve_url()

    def _entry(name):
        # `hosted_vllm/` (NOT generic `openai/`) is the provider that lifts
        # vLLM's `reasoning_content` onto the streamed delta. The generic
        # openai/ transformer does not, so the Anthropic /v1/messages adapter's
        # reasoning->thinking branch never fired and transcripts captured 0
        # thinking blocks (needed for P4 LoRA targets). The adapter itself
        # already emits thinking blocks from reasoning_content in litellm 1.91 —
        # it just needs the vLLM provider to populate it. See FINDINGS
        # 2026-07-08 P4 recapture incident + research.
        return {
            "model_name": name,
            "litellm_params": {
                "model": f"hosted_vllm/{name}",
                "api_base": f"{serve_url}/v1",
                "api_key": os.environ["VLLM_API_KEY"],
            },
            # Capability tag: gates request-side thinking->reasoning_effort
            # mapping + token accounting. Necessary but not sufficient on its own
            # (the provider prefix above is the load-bearing change).
            "model_info": {"supports_reasoning": True},
        }

    config = {
        # Both the base and the P4 LoRA are served on one vLLM endpoint (serve()
        # --lora-modules), so the proxy exposes both. The re-gate runs the tuned
        # `ornith-lora` paired against `ornith-35b` on the same stack.
        "model_list": [_entry("ornith-35b"), _entry("ornith-lora")],
        # NB: do NOT set merge_reasoning_content_in_choices — that folds <think>
        # back into `content` (reintroduces the leak the parser removes). We want
        # reasoning as SEPARATE thinking blocks, not merged text.
        "litellm_settings": {"drop_params": True},
    }
    import yaml
    Path("/root/litellm_config.yaml").write_text(yaml.safe_dump(config))
    subprocess.Popen(
        ["litellm", "--config", "/root/litellm_config.yaml",
         "--port", "4000", "--host", "0.0.0.0"],
        env={**os.environ, "LITELLM_MASTER_KEY": os.environ["PROXY_MASTER_KEY"]},
    )


def _web_url(fn_name: str) -> str:
    """Resolve a deployed function's web URL (Modal API drift-tolerant)."""
    import modal as _modal
    fn = _modal.Function.from_name("ornith-harness", fn_name)
    for attr in ("get_web_url", "web_url"):
        v = getattr(fn, attr, None)
        url = v() if callable(v) else v
        if url:
            return url.rstrip("/")
    raise SystemExit(f"could not resolve {fn_name}() web URL — is it deployed?")


def _proxy_url() -> str:
    return _web_url("proxy")


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
    return _web_url("serve")


# Slim image (NOT vllm_image) so fetching the endpoint key is a cheap cold start,
# not a multi-minute GPU-image boot.
@app.function(image=modal.Image.debian_slim(), secrets=[ENDPOINT_KEYS])
def _endpoint_key() -> str:
    return os.environ["VLLM_API_KEY"]


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


def _chat(url: str, messages, tools=None, tool_choice=None, max_tokens=1024,
          api_key: str = "", model: str = "ornith-35b") -> dict:
    import json
    import urllib.request
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    headers = {"Content-Type": "application/json"}
    if api_key:  # serve()'s /v1/* is Bearer-gated by --api-key
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


@app.local_entrypoint()
def smoke(quick: bool = False, model: str = "ornith-35b"):
    """Phase 0 gate (G1). Exits nonzero on any failure so it can gate CI.

    Asserts, against the deployed endpoint (the same OpenAI schema the
    Anthropic proxy renames field-for-field):
      (1) tool_calls parse into a valid Anthropic tool_use block,
      (2) no response content contains '<think>' / '</think>',
      (3) all trivial tasks are answered correctly.

    --quick runs 3 trivials instead of 20 — a cheap liveness check that still
    cold-boots the GPU but skips 17 serial generations. The G1 GATE is the FULL
    run (status.py uses it); use --quick only for a manual "is it alive?".

    --model selects the served model: 'ornith-35b' (base, the G1 gate) or
    'ornith-lora' (the P4 adapter re-gate — same three assertions on the tuned
    model before the paired dev run).
    """
    import json
    import time

    trivials = TRIVIALS[:3] if quick else TRIVIALS
    t0 = time.time()
    url = _serve_url()
    print(f"smoke: endpoint {url} ({'quick' if quick else 'full'}) model={model}")
    _wait_healthy(url)  # /health is unauthenticated
    print("smoke: endpoint healthy")
    key = _endpoint_key.remote()  # serve()'s /v1/* needs the Bearer key

    failures = []
    leaks = 0

    # (1) + (3): trivials, checking correctness and <think> leakage together.
    passed = 0
    for i, (prompt, expect) in enumerate(trivials, 1):
        try:
            resp = _chat(url, [{"role": "user", "content": prompt}], api_key=key,
                         model=model)
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
    print(f"smoke: trivials {passed}/{len(trivials)} correct; think-leaks {leaks}")

    # (2)/(1): forced tool call → schema-clean → maps to Anthropic tool_use.
    tool_ok = False
    try:
        resp = _chat(
            url,
            [{"role": "user", "content": "Use the add tool to compute 12345 + 67890."}],
            tools=[ADD_TOOL],
            tool_choice="auto",
            api_key=key,
            model=model,
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

    # Ledger the GPU time this smoke used (boot + generations) so the budget
    # hook — which reads the ledger — isn't blind to it. Smoke boots were
    # previously un-ledgered, so repeated smokes/status checks accumulated real
    # H100 $ the $150 cap never saw. Conservative: charge the full wall time as
    # GPU-seconds. Recorded even on failure (the GPU time was spent regardless).
    elapsed = time.time() - t0
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sweep_stats
        stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
        mtag = "" if model == "ornith-35b" else f"{_slug(model)}-"
        sweep_stats.append_ledger(
            Path(__file__).resolve().parent.parent / "findings" / "cost-ledger.csv",
            f"smoke-{mtag}{'quick-' if quick else ''}{stamp}",
            f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%S}",
            round(elapsed, 1), round(elapsed / 3600 * H100_USD_PER_HOUR, 4))
    except Exception as e:
        print(f"smoke: (ledger append skipped: {e})")

    ok = passed == len(trivials) and leaks == 0 and tool_ok
    if not ok:
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(
            f"smoke: FAILED ({passed}/{len(trivials)} trivials, "
            f"{leaks} think-leaks, tool_ok={tool_ok})"
        )
    print(f"smoke: PASS ({passed}/{len(trivials)} trivials, 0 think-leaks, "
          "schema-clean tool call)")


# run_trial only orchestrates sandboxes (the task container is where code
# actually runs); it needs no GPU/vLLM, so a slim image keeps its cold start
# small. trial_logic ships as local source so the pure verdict/parse/env logic
# is importable in the container.
# P8: run_trial imports selection_analysis in-container to read a just-finished
# session's transcript for the worker's own VERIFY signal (the attempts stop
# condition). selection_analysis pulls in export_trajectories + sweep_stats at
# module load, and both selection_analysis and export_trajectories `import yaml`
# at top level, so the closure needs pyyaml + all three local modules shipped —
# not just the two the P8 pre-registration named (it undercounted the closure).
harness_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyyaml")
    .add_local_python_source(
        "trial_logic", "selection_analysis", "export_trajectories", "sweep_stats"
    )
)


def _create_sandbox(app_handle, **kwargs):
    """Sandbox.create with one 30s-backoff retry — sandbox creation is the most
    common transient failure (image pull, scheduler). Raises on final failure;
    the caller maps that to an invalid verdict, never a 'fail'."""
    last = None
    for attempt in range(2):
        try:
            return modal.Sandbox.create(app=app_handle, **kwargs)
        except Exception as e:  # transient scheduler/pull error
            last = e
            if attempt == 0:
                time.sleep(30)
    raise last


def _sb_write(sb, path: str, content) -> None:
    """Write a file into a running sandbox (text or bytes), creating parents."""
    sb.exec("mkdir", "-p", str(PurePosixPath(path).parent)).wait()
    mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
    with sb.open(path, mode) as f:
        f.write(content)


def _write_verdict(out_dir: Path, verdict: str, exit_code, stderr_tail: str,
                   base_sha: str, reason: str) -> None:
    (out_dir / "verdict.json").write_text(json.dumps({
        "verdict": verdict, "exit_code": exit_code,
        "stderr_tail": (stderr_tail or "")[-2000:],
        "base_sha": base_sha, "reason": reason,
    }, indent=2))


@app.function(image=harness_image, volumes={"/runs": runs_vol},
              timeout=120 * MINUTES, max_containers=24, secrets=[ENDPOINT_KEYS])
def run_trial(task_spec: dict, variant_config_tar: bytes, trial_idx: int,
              run_id: str, worker: dict, attempts: int = 1) -> dict:
    """One task x one trial, inside the task's pinned container.

    Invariants this function owns (not the model, not the skill):
      - PHASE A (worker): derive the task image + Node/Claude-Code, materialize
        variant claude-config/ as .claude/ INSIDE the task workspace only, write
        task_spec['verify'] to VERIFY.txt and the description to TASK.md, then
        run the worker (Claude Code CLI against `worker`'s endpoint). The
        worker's change is captured as a git diff that EXCLUDES the harness
        scaffolding (.claude, VERIFY.txt, TASK.md).
      - PHASE B (verdict, hidden-tests contract, see tasks/schema.md): in a
        FRESH raw task container with the network blocked, `git apply` the
        worker diff, then `git apply` task_spec['hidden_tests'] (injected ONLY
        here — the worker never sees the tests, corpus-curator item 6: no oracle
        leakage), then run task_spec['verify']; its exit code is the verdict.
      - records tokens, gpu_seconds, api_usd, wall_clock, transcript, verdict;
      - an execution error is verdict='invalid', never 'fail'.
    """
    import trial_logic as tl

    started = time.time()
    task_id = task_spec["id"]
    timeout_s = int(task_spec.get("timeout_s", 1800))
    description = task_spec.get("description", "")
    out_dir = Path("/runs") / run_id / task_id / f"trial{trial_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.jsonl"
    app_handle = modal.App.lookup("ornith-harness")

    # Attribution flag (F7): all workers are GPU-attributed (shared vLLM fleet,
    # billed by wall time) since the hosted-worker path was removed 2026-07-09.
    # The explicit flag is kept so attribution never infers from base_url.
    gpu_attributed = bool(worker.get("gpu_attributed"))

    def _result(verdict, reason, usage=None, timed_out=False, error=None):
        usage = usage or {}
        tokens_out = usage.get("tokens_out", 0)
        res = {
            "task": task_id, "trial": trial_idx,
            "verdict": verdict, "reason": reason,
            "wall_clock_s": round(time.time() - started, 2),
            "tokens_in": usage.get("tokens_in", 0),
            "tokens_out": tokens_out,
            # api_usd is a legacy field kept for ledger/stats schema stability;
            # always 0.0 now (no hosted workers). Claude Code self-reports
            # total_cost_usd from its own pricing table even against the proxy,
            # so recording it for the ornith endpoint would be phantom $.
            "gpu_seconds": (tokens_out / tl.AGG_TOK_PER_S) if gpu_attributed else 0.0,
            "api_usd": 0.0 if gpu_attributed else usage.get("api_usd", 0.0),
            "transcript_path": str(transcript_path.relative_to("/runs")),
            "timed_out": timed_out,
            "num_turns": usage.get("num_turns", 0),
            # P8 wrapper telemetry: how many sessions actually ran and why the
            # loop stopped. At attempts=1 these are (0 or 1, "single_attempt");
            # they never alter the existing fields, so attempts=1 is unchanged.
            "n_attempts": n_attempts_run,
            "stop_reason": stop_reason,
        }
        if error:
            res["error"] = error
        runs_vol.commit()
        return res

    # Oracle-leak guard backstop (also enforced at build time in _tar_config):
    # the variant config must never carry the hidden tests. Basename match.
    for m in tarfile.open(fileobj=io.BytesIO(variant_config_tar), mode="r:gz").getmembers():
        if PurePosixPath(m.name).name == "tests.patch":
            raise RuntimeError(
                f"variant config contains {m.name!r} — refusing (oracle leak guard)")

    # The worker authenticates to the in-app proxy with PROXY_MASTER_KEY (from
    # run_trial's own secret env); fill it into the worker before building the
    # exec env. (Hosted workers and their separate Anthropic-key secret were
    # removed 2026-07-09 — every worker is proxy-routed now.)
    worker = {**worker, "api_key": os.environ["PROXY_MASTER_KEY"]}
    exec_env = tl.worker_env(worker)

    worker_img = modal.Image.from_registry(task_spec["image"]).run_commands(
        *tl.node_claude_install_cmds())
    raw_img = modal.Image.from_registry(task_spec["image"])

    base_sha = ""
    worker_diff = None
    usage = {}
    # acc folds usage across ALL sessions this trial runs (cost honesty: the
    # wrapper multiplies inference). gpu_seconds is derived once from the summed
    # tokens_out in _result, so it is not summed here.
    acc = {"tokens_in": 0, "tokens_out": 0, "num_turns": 0, "api_usd": 0.0}
    timed_out = False
    n_attempts_run = 0
    # Default for pre-session invalids and the attempts=1 no-op path; the loop
    # overwrites it with a real reason once >1 session runs.
    stop_reason = "single_attempt" if attempts == 1 else "attempts_exhausted"

    def _run_claude_session(sb, prompt, tpath):
        """Stream ONE `claude -p` session inside the ALREADY-created sandbox
        `sb`, writing stream-json to `tpath`. Returns (usage, sess_timed_out,
        lines). The workspace in `sb` PERSISTS after return — the caller owns
        `sb` and its termination; sequential calls on one `sb` are exactly the
        inference-time persistence channel P8 reconstructs (nothing is reset
        between sessions). The timeout stays timeout_s PER SESSION."""
        lines: list[str] = []
        buf = ""
        # modal 1.5.1: an exec deadline ends stdout iteration SILENTLY (no
        # exception) and makes wait() return -1. Detect timeout from that, not
        # from a (never-raised) ExecTimeoutError. Keep a UnicodeDecodeError guard
        # as cheap insurance against odd bytes in the text stream.
        exec_start = time.time()
        # Run claude under bash with stdin from /dev/null: an open non-TTY stdin
        # makes `claude -p` block forever before any output. The prompt goes
        # through an env var (WORKER_PROMPT), never the shell command string, so
        # arbitrary issue text can't break quoting. `exec` replaces bash so
        # signals/exit flow straight to the CLI.
        proc = sb.exec(
            "bash", "-lc",
            'exec claude -p "$WORKER_PROMPT" --output-format stream-json '
            "--verbose --dangerously-skip-permissions --max-turns 150 "
            "< /dev/null",
            env={**exec_env, "WORKER_PROMPT": prompt},
            timeout=timeout_s, workdir="/testbed")
        with open(tpath, "w") as tfp:
            try:
                for chunk in proc.stdout:
                    buf += chunk
                    while "\n" in buf:
                        ln, buf = buf.split("\n", 1)
                        tfp.write(ln + "\n")
                        lines.append(ln)
            except UnicodeDecodeError:  # odd bytes in transcript -> stream end
                pass
            finally:
                if buf:
                    tfp.write(buf + "\n")
                    lines.append(buf)
        rc = proc.wait()
        elapsed = time.time() - exec_start
        sess_timed_out = (rc == -1 and elapsed >= timeout_s - 2)
        # Capture worker stderr (bounded); the LAST session to run keeps the
        # file (same path/purpose as the single-shot path) for post-hoc
        # diagnosis of invalid trials.
        try:
            (out_dir / "worker.stderr.log").write_text(
                (proc.stderr.read() or "")[-4000:])
        except Exception:
            pass
        return tl.parse_stream_json(lines), sess_timed_out, lines

    def _stop_after(sess_usage, sess_timed_out, lines):
        """Loop control after a session (attempts>1 only). Returns
        (stop, reason): stop on the worker's OWN last VERIFY run passing
        (reason 'self_verify_pass', the B-validated oracle-blind signal) or on a
        timeout (reason None; the timed_out field already records it). A dead /
        self-errored session is a failed attempt -> keep going if attempts
        remain, else fall through to Phase B."""
        if sess_timed_out:
            return True, None
        if not sess_usage.get("found_result") or sess_usage.get("is_error"):
            return False, None
        import selection_analysis as sa  # in-container via add_local_python_source
        wv = sa.detect_worker_verify(lines, task_spec.get("verify") or "")
        if tl.should_stop_attempts(wv["state"]):
            return True, "self_verify_pass"
        return False, None

    # --- Phase A: up to `attempts` sequential sessions in ONE sandbox ---------
    sb = None
    try:
        # Attempt 1: fresh sandbox + scaffold, with the historic single
        # recreate-retry on an abnormal (no-result, non-timeout) worker death.
        # That retry RECREATES the sandbox — legitimate ONLY here, before any
        # carried workspace exists. Attempts >=2 reuse this sandbox and NEVER
        # recreate (workspace persistence is the whole point of P8).
        for phase_attempt in range(2):
            try:
                sb = _create_sandbox(app_handle, image=worker_img,
                                     timeout=timeout_s + 600, workdir="/testbed",
                                     cpu=2)
            except Exception as e:
                # _create_sandbox already exhausted its internal retry budget.
                return _result(
                    tl.classify("worker_sandbox_create_failed", "exhausted"),
                    "worker_sandbox_create_failed",
                    error=f"sandbox create failed: {e}")

            # git rev-parse must succeed and yield a SHA — Phase B diffs against
            # it, so an unresolved base is an execution error, not a fail.
            sha_proc = sb.exec("git", "rev-parse", "HEAD", workdir="/testbed")
            base_sha = sha_proc.stdout.read().strip()
            if sha_proc.wait() != 0 or not base_sha:
                return _result(tl.classify("base_sha_unresolved"),
                               "base_sha_unresolved", usage=acc,
                               error="git rev-parse HEAD failed or empty")

            _sb_write(sb, "/tmp/cfg.tgz", variant_config_tar)
            mrc = sb.exec("bash", "-lc",
                          "mkdir -p /testbed/.claude /tmp/cfgx && "
                          "tar -xzf /tmp/cfg.tgz -C /tmp/cfgx && "
                          "cp -a /tmp/cfgx/claude-config/. /testbed/.claude/").wait()
            if mrc != 0:
                return _result(tl.classify("scaffold_materialize_failed"),
                               "scaffold_materialize_failed", usage=acc,
                               error=f"scaffold materialize exit {mrc}")
            _sb_write(sb, "/testbed/VERIFY.txt", task_spec["verify"])
            _sb_write(sb, "/testbed/TASK.md", description)

            usage, timed_out, lines = _run_claude_session(
                sb, tl.build_attempt_prompt(description, 1, attempts),
                transcript_path)
            n_attempts_run = 1
            tl.accumulate_usage(acc, usage)

            # No terminating result line and not a timeout => the CLI died
            # abnormally; recreate a FRESH sandbox once and retry the first
            # session (nothing carried is lost — no workspace exists yet).
            if not usage["found_result"] and not timed_out:
                if phase_attempt == 0:
                    try:
                        sb.terminate()
                    except Exception:
                        pass
                    sb = None
                    acc = {"tokens_in": 0, "tokens_out": 0,
                           "num_turns": 0, "api_usd": 0.0}
                    n_attempts_run = 0
                    continue
                # Retry exhausted. attempts=1: historic terminal invalid.
                # attempts>1: one failed attempt — fall through so sessions 2..n
                # still run (the recreated sandbox is fresh).
                if attempts == 1:
                    return _result(
                        tl.classify("worker_no_result_line", "exhausted"),
                        "worker_no_result_line", usage=acc, timed_out=timed_out,
                        error="worker produced no result line")

            # Ran to completion but the CLI self-reported an error (max turns,
            # API error, ...). attempts=1: historic terminal invalid. attempts>1:
            # a failed attempt (handled by _stop_after), never a hard return.
            if usage["found_result"] and usage["is_error"] and attempts == 1:
                return _result(tl.classify("worker_reported_error"),
                               "worker_reported_error", usage=acc,
                               timed_out=timed_out, error=usage["subtype"])
            break

        # Sessions 2..attempts reuse the SAME sandbox (workspace persists). Stop
        # early on a worker self-VERIFY pass or a timeout; else run them all.
        if attempts > 1:
            stop, reason = _stop_after(usage, timed_out, lines)
            stop_reason = reason or "attempts_exhausted"
            if not stop:
                for k in range(2, attempts + 1):
                    tpath = out_dir / f"transcript.attempt{k}.jsonl"
                    usage, timed_out, lines = _run_claude_session(
                        sb, tl.build_attempt_prompt(description, k, attempts),
                        tpath)
                    n_attempts_run = k
                    tl.accumulate_usage(acc, usage)
                    stop, reason = _stop_after(usage, timed_out, lines)
                    stop_reason = reason or "attempts_exhausted"
                    if stop:
                        break

        # Capture the worker's change on the FINAL workspace, excluding harness
        # scaffolding. Runs ONCE, after the loop — Phase B verdicts this state.
        arc = sb.exec("git", "add", "-A", "--",
                      ":(exclude).claude", ":(exclude)VERIFY.txt",
                      ":(exclude)TASK.md", workdir="/testbed").wait()
        if arc != 0:
            return _result(tl.classify("worker_diff_stage_failed"),
                           "worker_diff_stage_failed", usage=acc,
                           timed_out=timed_out, error=f"git add exit {arc}")
        # Binary-safe: text-mode streams decode UTF-8 strict and would raise on
        # a binary diff hunk. Read bytes; _sb_write/write_bytes handle them.
        dproc = sb.exec("git", "diff", "--binary", "--cached", base_sha,
                        workdir="/testbed", text=False)
        worker_diff = dproc.stdout.read()
        dproc.wait()
        (out_dir / "worker.diff").write_bytes(worker_diff)
    finally:
        if sb is not None:
            try:
                sb.terminate()
            except Exception:
                pass

    if not worker_diff or not worker_diff.strip():
        return _result(tl.classify("empty_diff"), "empty_diff", usage=acc,
                       timed_out=timed_out)

    # --- Phase B: verdict, in a fresh raw container with the network blocked ---
    vsb = None
    try:
        try:
            vsb = _create_sandbox(app_handle, image=raw_img, block_network=True,
                                  timeout=timeout_s + 300, workdir="/testbed", cpu=2)
        except Exception as e:
            return _result(tl.classify("verdict_sandbox_create_failed"),
                           "verdict_sandbox_create_failed",
                           usage=acc, timed_out=timed_out,
                           error=f"verdict sandbox create failed: {e}")

        _sb_write(vsb, "/tmp/worker.diff", worker_diff)
        _sb_write(vsb, "/tmp/tests.patch", task_spec.get("hidden_tests_content") or "")

        rc = vsb.exec("git", "apply", "--whitespace=nowarn", "/tmp/worker.diff",
                      workdir="/testbed").wait()
        if rc != 0:
            v = tl.classify("worker_diff_apply_failed")
            _write_verdict(out_dir, v, rc, "", base_sha, "worker_diff_apply_failed")
            return _result(v, "worker_diff_apply_failed", usage=acc,
                           timed_out=timed_out)

        # Reset the files the hidden tests touch back to base BEFORE applying
        # them: the worker may have edited test files (it did on the first real
        # trial), which both conflicts with the patch and would let a worker
        # game the tests. Tests are authoritative (SWE-bench semantics) — the
        # worker's source fix is kept, its test edits are discarded.
        test_files = tl.patch_target_files(task_spec.get("hidden_tests_content") or "")
        if test_files:
            vsb.exec("git", "checkout", base_sha, "--", *test_files,
                     workdir="/testbed").wait()

        rc = vsb.exec("git", "apply", "/tmp/tests.patch", workdir="/testbed").wait()
        if rc != 0:
            v = tl.classify("hidden_tests_apply_failed")
            _write_verdict(out_dir, v, rc, "", base_sha, "hidden_tests_apply_failed")
            return _result(v, "hidden_tests_apply_failed", usage=acc,
                           timed_out=timed_out)

        # modal 1.5.1: a verify exec deadline makes wait() return -1 silently
        # (no ExecTimeoutError). Read stderr bytes (binary-safe) and detect the
        # deadline from rc==-1 + elapsed. text=False -> decode ourselves.
        v_start = time.time()
        vproc = vsb.exec("bash", "-lc", task_spec["verify"],
                         timeout=timeout_s, workdir="/testbed", text=False)
        err_tail = vproc.stderr.read().decode("utf-8", errors="replace")
        exit_code = vproc.wait()
        v_elapsed = time.time() - v_start
        if exit_code == -1 and v_elapsed >= timeout_s - 2:
            reason, exit_code = "verify_timeout", None
        elif exit_code == 0:
            reason = "verify_exit_zero"
        else:
            reason = "verify_exit_nonzero"
        verdict = tl.classify(reason)
        _write_verdict(out_dir, verdict, exit_code, err_tail, base_sha, reason)
        return _result(verdict, reason, usage=acc, timed_out=timed_out)
    finally:
        if vsb is not None:
            try:
                vsb.terminate()
            except Exception:
                pass


H100_USD_PER_HOUR = 3.95  # Modal H100 list rate; the ledger is the source of truth


def _load_spec(task_dir: Path) -> dict:
    """task.yaml -> the dict run_trial needs, with the hidden tests inlined."""
    import yaml
    d = yaml.safe_load((task_dir / "task.yaml").read_text())
    ht = d.get("hidden_tests")
    d["hidden_tests_content"] = (task_dir / ht).read_text() if ht else None
    return d


def _tar_config(variant_dir: Path) -> bytes:
    """Tar the variant's claude-config/ for materialization inside the task
    container. Structural invariant: only claude-config/ ships — never this
    repo's own .claude/."""
    import io
    import tarfile
    cfg = variant_dir / "claude-config"
    if not cfg.is_dir():
        raise SystemExit(f"variant has no claude-config/: {variant_dir}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(cfg, arcname="claude-config")
    payload = buf.getvalue()
    # Oracle-leak guard at build time: fail fast locally if the variant config
    # carries the hidden tests. Exact basename match (run_trial keeps a backstop).
    for m in tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz").getmembers():
        if PurePosixPath(m.name).name == "tests.patch":
            raise SystemExit(
                f"variant config contains {m.name!r} — refusing (oracle leak guard)")
    return payload


def _parent_per_task(parent: str | None, root: Path) -> dict | None:
    """Latest usable parent-variant run summary -> its per_task block, for paired
    stats. Selection is by SUMMARY FIELDS, not run_id string luck: iterate
    candidates newest-first and skip any that were partial runs or used a
    non-base worker (a LoRA-arm or partial baseline is not a valid parent to
    pair the subject against). The run_id suffixes stay for humans."""
    if not parent:
        return None
    runs = sorted((root / "experiments" / "runs").glob(f"*-{parent}*/summary.json"))
    for path in reversed(runs):
        summ = json.loads(path.read_text())
        if summ.get("variant") != parent:  # broad glob may catch sibling variants
            continue
        if summ.get("partial"):
            continue
        if summ.get("worker_model") not in (None, "ornith-35b"):
            continue
        return {t: {"valid": v["valid"], "pass_rate": v["pass_rate"]}
                for t, v in summ.get("per_task", {}).items()}
    return None


def _slug(s: str) -> str:
    """Filesystem-safe slug for a model name (for run_id suffixes)."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def _make_worker(worker_model: str) -> dict:
    """worker dict consumed by run_trial. Both served models (the base and the
    P4 LoRA) route through the in-app vLLM+LiteLLM proxy stack, GPU-attributed;
    the api_key is filled in run_trial from the proxy's own secret env. Same
    routing; only the served-model name differs.

    Hosted reference workers were removed by the 2026-07-09 operator redirect
    (all comparisons are internal, base-vs-tuned) — reject anything else so a
    stale flag can't silently reach an external API."""
    if worker_model in ("ornith-35b", "ornith-lora"):
        return {"model": worker_model, "small_model": worker_model,
                "base_url": _proxy_url(), "api_key": None,
                "gpu_attributed": True}
    raise SystemExit(
        f"unknown worker_model {worker_model!r}: hosted reference workers were "
        "removed (operator redirect 2026-07-09); use 'ornith-35b' or "
        "'ornith-lora'")


def _probe_proxy(url: str) -> None:
    """Best-effort liveliness probe of the LiteLLM proxy. Its /health/liveliness
    is public (LiteLLM leaves it unauthenticated), so no key is needed here —
    run_sweep does not hold the proxy key locally. Raises on non-200."""
    import urllib.request
    with urllib.request.urlopen(f"{url}/health/liveliness", timeout=15) as r:
        if r.status != 200:
            raise RuntimeError(f"proxy liveliness status {r.status}")


@app.local_entrypoint()
def run_sweep(variant: str, trials: int = 5, split: str = "dev",
              worker_model: str = "ornith-35b", tasks: str = "",
              attempts: int = 1):
    """Fan a variant out across the task set; write runs/<run_id>/ and append
    the cost ledger. Holdout runs require the operator's .holdout-unlocked
    flow upstream — this entrypoint refuses split='holdout' unless the flag
    file exists, as defense in depth.

    worker_model selects the served model: 'ornith-35b' (the base) or
    'ornith-lora' (the P4 adapter); both run on the in-app vLLM+proxy stack,
    GPU-budget-gated. Hosted reference workers were removed 2026-07-09.
    tasks: optional comma-separated task-id filter (partial run).
    attempts: P8 native-loop wrapper — up to N sequential Claude Code sessions
    per trial in ONE persistent sandbox, early-stopping on the worker's own
    VERIFY pass (default 1 = today's single-shot; P8 uses 3)."""
    import sys
    import time as _time
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "infra"))
    import sweep_stats as stats

    if split == "holdout" and not (root / ".holdout-unlocked").exists():
        raise SystemExit("holdout sealed: operator must create .holdout-unlocked")
    if trials < 3:
        raise SystemExit("trials < 3: paired stats need >=3 (PLAN decision rules)")
    if not 1 <= attempts <= 5:
        raise SystemExit("attempts must be in [1, 5] (P8 uses 3)")

    variant_dir = root / "experiments" / "variants" / variant
    manifest = variant_dir / "manifest.yaml"
    if not manifest.exists():
        raise SystemExit(f"no such variant: {variant}")
    import yaml
    parent = yaml.safe_load(manifest.read_text()).get("parent")

    is_ornith = worker_model in ("ornith-35b", "ornith-lora")

    # glob yields the task.yaml FILES; _load_spec wants the task DIRECTORY.
    task_dirs = sorted(p.parent for p in (root / "tasks" / split).glob("*/task.yaml"))
    if not task_dirs:
        raise SystemExit(f"no tasks in tasks/{split}/")
    specs = [_load_spec(d) for d in task_dirs]

    partial = bool(tasks.strip())
    if partial:
        wanted = {t.strip() for t in tasks.split(",") if t.strip()}
        specs = [s for s in specs if s["id"] in wanted]
        if not specs:
            raise SystemExit(f"no tasks in tasks/{split}/ match {sorted(wanted)}")

    # A task with no hidden tests cannot be verdicted (Phase B would git-apply an
    # empty patch -> invalid), which silently shrinks the valid set and corrupts
    # the stats. Fail fast instead. Phase B keeps the apply-failure backstop.
    missing = [s["id"] for s in specs if not s.get("hidden_tests_content")]
    if missing:
        raise SystemExit(f"tasks missing hidden_tests_content: {missing}")

    cfg_tar = _tar_config(variant_dir)
    worker = _make_worker(worker_model)  # rejects unknown worker models

    ts = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    run_id = f"{ts}-{variant}" if is_ornith else f"{ts}-{variant}-{_slug(worker_model)}"
    if partial:
        run_id += "-partial"

    print(f"[sweep] {run_id}: {len(specs)} tasks x {trials} trials "
          f"({len(specs) * trials} trials), variant {variant} (parent {parent}) "
          f"worker {worker_model} attempts {attempts}")

    if is_ornith:
        _wait_healthy(_serve_url())
        try:
            _probe_proxy(_proxy_url())
        except Exception as e:  # best-effort; the per-trial calls will surface it
            print(f"[sweep] proxy health probe failed (continuing): {e}")

    work = [(spec, cfg_tar, t, run_id, worker, attempts)
            for spec in specs for t in range(trials)]
    prov = {s["id"]: s.get("provenance") for s in specs}

    sweep_start = _time.time()
    results = []
    # return_exceptions: a single trial raising (wrapped user-code exception)
    # must not sink the whole sweep — synthesize an invalid result for it.
    # order_outputs defaults True, so zip(work, results) stays aligned.
    for (spec, _cfg, t, _rid, _w, _a), r in zip(
            work, run_trial.starmap(work, return_exceptions=True)):
        if isinstance(r, Exception):
            r = {"task": spec["id"], "trial": t, "verdict": "invalid",
                 "reason": "trial_exception", "error": str(r)[:300],
                 "tokens_in": 0, "tokens_out": 0, "gpu_seconds": 0.0,
                 "api_usd": 0.0, "wall_clock_s": 0.0, "timed_out": False,
                 "num_turns": 0, "transcript_path": "",
                 "n_attempts": 0, "stop_reason": "trial_exception"}
        r["provenance"] = prov.get(r.get("task"))
        results.append(r)
    sweep_wall_s = _time.time() - sweep_start

    run_dir = root / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Write the raw trials BEFORE summarize so a summarize bug can't lose data.
    with (run_dir / "trials.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = stats.summarize(run_id, variant, parent, results,
                              _parent_per_task(parent, root), H100_USD_PER_HOUR,
                              worker_model=worker_model)
    # Recorded as a field (not inferred from the run_id suffix) so _parent_per_task
    # can exclude partial runs as parents by data, not by string luck (F11).
    summary["partial"] = partial
    stats.write_summary(run_dir, summary)

    # Ledger charges GPU time: the larger of summed per-trial attribution and
    # the sweep wall clock (ornith only — the vLLM fleet bills by wall time).
    gpu_seconds_ledger = max(
        sum(r.get("gpu_seconds", 0.0) for r in results),
        sweep_wall_s if is_ornith else 0.0)
    usd_ledger = gpu_seconds_ledger / 3600.0 * H100_USD_PER_HOUR
    stats.append_ledger(root / "findings" / "cost-ledger.csv", run_id,
                        f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%S}",
                        gpu_seconds_ledger, usd_ledger)

    c, p = summary["cost"], summary["paired_vs_parent"]
    print(f"[sweep] {run_id}: solved {summary['solved_tasks']}/{summary['valid_tasks']} "
          f"tasks (rate {summary['pass_rate_over_tasks']}) · paired vs {parent}: "
          f"+{p['wins']}/-{p['losses']}/={p['ties']} net {p['net_tasks']} · "
          f"${c['usd']} gpu + ${c.get('api_usd', 0.0)} api "
          f"({c['invalid_trials']} invalid) · runs/{run_id}/summary.json")


@app.local_entrypoint()
def run_one(task_id: str, variant: str = "v001-baseline",
            worker_model: str = "ornith-35b", attempts: int = 1):
    """One task x one trial, for exercising the runner end-to-end (the
    proof-of-one). Writes NOTHING to experiments/runs/ or the ledger — the trial
    still writes its own transcript/diff/verdict to the /runs volume — and prints
    the result dict as JSON. Refuses to reach into sealed holdout.

    attempts: P8 native-loop wrapper — up to N sequential sessions in one
    persistent sandbox (default 1 = single-shot)."""
    root = Path(__file__).resolve().parent.parent
    if not 1 <= attempts <= 5:
        raise SystemExit("attempts must be in [1, 5] (P8 uses 3)")
    variant_dir = root / "experiments" / "variants" / variant
    if not (variant_dir / "manifest.yaml").exists():
        raise SystemExit(f"no such variant: {variant}")

    spec = None
    for split in ("dev", "staging"):
        d = root / "tasks" / split / task_id
        if (d / "task.yaml").exists():
            spec = _load_spec(d)
            break
    if spec is None:
        raise SystemExit(f"task not found in dev/ or staging/: {task_id}")
    if not spec.get("hidden_tests_content"):
        raise SystemExit(f"task missing hidden_tests_content: {task_id}")

    cfg_tar = _tar_config(variant_dir)
    worker = _make_worker(worker_model)
    if worker_model in ("ornith-35b", "ornith-lora"):
        _wait_healthy(_serve_url())
    ts = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    result = run_trial.remote(spec, cfg_tar, 0, f"one-{ts}-{task_id}", worker,
                              attempts)
    print(json.dumps(result, indent=2))
