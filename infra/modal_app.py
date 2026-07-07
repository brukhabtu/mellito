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

# Static, non-secret master key for the in-app LiteLLM proxy. The proxy only
# fronts the (already private) vLLM endpoint; this key gates the proxy port, it
# is not a provider credential, so it lives in code by design.
PROXY_MASTER_KEY = "sk-ornith-harness"

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
        # P2 throughput: cudagraph + torch.compile (inductor->triton, no nvcc)
        # remove the per-decode-step Python/launch overhead that pinned eager
        # mode to ~15 tok/s single-stream. Cap capture sizes so cold-boot graph
        # capture stays bounded (the reason eager was used at G1). Sampler and
        # GDN-prefill stay on their nvcc-free backends (set above).
        "--compilation-config",
        '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}',
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


@app.function(image=proxy_image, scaledown_window=300, timeout=60 * MINUTES)
# One proxy container fans many concurrent worker requests onto the shared vLLM
# fleet; match serve()'s concurrency so the proxy is never the bottleneck.
@modal.concurrent(max_inputs=64)
@modal.web_server(port=4000, startup_timeout=5 * MINUTES)
def proxy():
    """LiteLLM proxy exposing an Anthropic-format endpoint backed by vLLM.

    Resolves serve()'s deployed URL at boot and writes a one-model config
    routing Anthropic requests for `ornith-35b` to `openai/ornith-35b` at
    serve()/v1. drop_params tolerates Anthropic-only fields vLLM doesn't accept.
    """
    serve_url = _serve_url()
    config = {
        "model_list": [
            {
                "model_name": "ornith-35b",
                "litellm_params": {
                    "model": "openai/ornith-35b",
                    "api_base": f"{serve_url}/v1",
                    "api_key": "dummy",
                },
            }
        ],
        "litellm_settings": {"drop_params": True},
    }
    import yaml
    Path("/root/litellm_config.yaml").write_text(yaml.safe_dump(config))
    subprocess.Popen(
        ["litellm", "--config", "/root/litellm_config.yaml",
         "--port", "4000", "--host", "0.0.0.0"],
        env={**os.environ, "LITELLM_MASTER_KEY": PROXY_MASTER_KEY},
    )


def _proxy_url() -> str:
    import modal as _modal
    fn = _modal.Function.from_name("ornith-harness", "proxy")
    for attr in ("get_web_url", "web_url"):
        v = getattr(fn, attr, None)
        url = v() if callable(v) else v
        if url:
            return url.rstrip("/")
    raise SystemExit("proxy: could not resolve proxy() web URL — is it deployed?")


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


# run_trial only orchestrates sandboxes (the task container is where code
# actually runs); it needs no GPU/vLLM, so a slim image keeps its cold start
# small. trial_logic ships as local source so the pure verdict/parse/env logic
# is importable in the container.
harness_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "trial_logic"
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
              timeout=120 * MINUTES, max_containers=24)
def run_trial(task_spec: dict, variant_config_tar: bytes, trial_idx: int,
              run_id: str, worker: dict) -> dict:
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
    from modal.exception import ExecTimeoutError
    import trial_logic as tl

    started = time.time()
    task_id = task_spec["id"]
    timeout_s = int(task_spec.get("timeout_s", 1800))
    description = task_spec.get("description", "")
    out_dir = Path("/runs") / run_id / task_id / f"trial{trial_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.jsonl"
    app_handle = modal.App.lookup("ornith-harness")

    def _result(verdict, reason, usage=None, timed_out=False, error=None):
        usage = usage or {}
        tokens_out = usage.get("tokens_out", 0)
        res = {
            "task": task_id, "trial": trial_idx,
            "verdict": verdict, "reason": reason,
            "wall_clock_s": round(time.time() - started, 2),
            "tokens_in": usage.get("tokens_in", 0),
            "tokens_out": tokens_out,
            # GPU-seconds only attributed to the ornith path (shared vLLM
            # fleet); claude-* spend is API dollars, tracked separately.
            "gpu_seconds": (tokens_out / tl.AGG_TOK_PER_S) if worker.get("base_url") else 0.0,
            "api_usd": usage.get("api_usd", 0.0),
            "transcript_path": str(transcript_path.relative_to("/runs")),
            "timed_out": timed_out,
            "num_turns": usage.get("num_turns", 0),
        }
        if error:
            res["error"] = error
        runs_vol.commit()
        return res

    # Oracle-leak guard: the variant config must never carry the hidden tests.
    for m in tarfile.open(fileobj=io.BytesIO(variant_config_tar), mode="r:gz").getmembers():
        if "tests.patch" in m.name:
            raise RuntimeError(
                f"variant config contains {m.name!r} — refusing (oracle leak guard)")

    # claude-* workers need a real Anthropic key; it lives in a Modal secret and
    # is injected into the sandbox container env (not the orchestrator), so
    # ornith-only runs need no secret at all. The exec env drops its placeholder
    # ANTHROPIC_API_KEY so the container's real key wins.
    worker_secrets = None
    exec_env = tl.worker_env(worker)
    if not worker.get("base_url"):
        worker_secrets = [modal.Secret.from_name(
            "anthropic-api-key", required_keys=["ANTHROPIC_API_KEY"])]
        exec_env.pop("ANTHROPIC_API_KEY", None)

    worker_img = modal.Image.from_registry(task_spec["image"]).run_commands(
        *tl.node_claude_install_cmds())
    raw_img = modal.Image.from_registry(task_spec["image"])

    base_sha = ""
    worker_diff = None
    usage = {}
    timed_out = False

    # --- Phase A: worker (with one full-phase retry on a missing result line) ---
    for phase_attempt in range(2):
        sb = None
        try:
            try:
                sb = _create_sandbox(app_handle, image=worker_img,
                                     timeout=timeout_s + 600, workdir="/testbed",
                                     cpu=2, secrets=worker_secrets)
            except Exception as e:
                return _result("invalid", "worker_sandbox_create_failed",
                               error=f"sandbox create failed: {e}")

            base_sha = sb.exec("git", "rev-parse", "HEAD",
                               workdir="/testbed").stdout.read().strip()

            _sb_write(sb, "/tmp/cfg.tgz", variant_config_tar)
            sb.exec("bash", "-lc",
                    "mkdir -p /testbed/.claude /tmp/cfgx && "
                    "tar -xzf /tmp/cfg.tgz -C /tmp/cfgx && "
                    "cp -a /tmp/cfgx/claude-config/. /testbed/.claude/").wait()
            _sb_write(sb, "/testbed/VERIFY.txt", task_spec["verify"])
            _sb_write(sb, "/testbed/TASK.md", description)

            lines: list[str] = []
            timed_out = False
            buf = ""
            proc = sb.exec(
                "claude", "-p", description,
                "--output-format", "stream-json", "--verbose",
                "--dangerously-skip-permissions", "--max-turns", "150",
                env=exec_env, timeout=timeout_s, workdir="/testbed")
            with open(transcript_path, "w") as tfp:
                try:
                    for chunk in proc.stdout:
                        buf += chunk
                        while "\n" in buf:
                            ln, buf = buf.split("\n", 1)
                            tfp.write(ln + "\n")
                            lines.append(ln)
                except ExecTimeoutError:  # worker ran past the task budget
                    timed_out = True
                finally:
                    if buf:
                        tfp.write(buf + "\n")
                        lines.append(buf)

            usage = tl.parse_stream_json(lines)

            # No terminating result line and not a timeout => the CLI died
            # abnormally; retry the whole worker phase once, then invalid.
            if not usage["found_result"] and not timed_out:
                if phase_attempt == 0:
                    continue
                return _result("invalid", "worker_no_result_line", usage=usage,
                               timed_out=timed_out,
                               error="worker produced no result line")

            # Capture the worker's change, excluding harness scaffolding.
            sb.exec("git", "add", "-A", "--",
                    ":(exclude).claude", ":(exclude)VERIFY.txt", ":(exclude)TASK.md",
                    workdir="/testbed").wait()
            dproc = sb.exec("git", "diff", "--binary", "--cached", base_sha,
                            workdir="/testbed")
            worker_diff = dproc.stdout.read()
            (out_dir / "worker.diff").write_text(worker_diff)
            break
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass

    if not worker_diff or not worker_diff.strip():
        return _result("fail", "empty_diff", usage=usage, timed_out=timed_out)

    # --- Phase B: verdict, in a fresh raw container with the network blocked ---
    vsb = None
    try:
        try:
            vsb = _create_sandbox(app_handle, image=raw_img, block_network=True,
                                  timeout=timeout_s + 300, workdir="/testbed", cpu=2)
        except Exception as e:
            return _result("invalid", "verdict_sandbox_create_failed",
                           usage=usage, timed_out=timed_out,
                           error=f"verdict sandbox create failed: {e}")

        _sb_write(vsb, "/tmp/worker.diff", worker_diff)
        _sb_write(vsb, "/tmp/tests.patch", task_spec.get("hidden_tests_content") or "")

        rc = vsb.exec("git", "apply", "--whitespace=nowarn", "/tmp/worker.diff",
                      workdir="/testbed").wait()
        if rc != 0:
            _write_verdict(out_dir, "invalid", rc, "", base_sha, "worker_diff_apply_failed")
            return _result("invalid", "worker_diff_apply_failed", usage=usage,
                           timed_out=timed_out)

        rc = vsb.exec("git", "apply", "/tmp/tests.patch", workdir="/testbed").wait()
        if rc != 0:
            _write_verdict(out_dir, "invalid", rc, "", base_sha, "hidden_tests_apply_failed")
            return _result("invalid", "hidden_tests_apply_failed", usage=usage,
                           timed_out=timed_out)

        vproc = vsb.exec("bash", "-lc", task_spec["verify"],
                         timeout=timeout_s, workdir="/testbed")
        try:
            err_tail = vproc.stderr.read()
            exit_code = vproc.wait()
            verdict, reason = ("pass", "verify_exit_zero") if exit_code == 0 \
                else ("fail", "verify_exit_nonzero")
        except ExecTimeoutError:
            verdict, reason, exit_code, err_tail = "fail", "verify_timeout", None, ""
        _write_verdict(out_dir, verdict, exit_code, err_tail, base_sha, reason)
        return _result(verdict, reason, usage=usage, timed_out=timed_out)
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
    return buf.getvalue()


def _parent_per_task(parent: str | None, root: Path) -> dict | None:
    """Latest parent-variant run summary -> its per_task block, for paired stats."""
    if not parent:
        return None
    runs = sorted((root / "experiments" / "runs").glob(f"*-{parent}/summary.json"))
    if not runs:
        return None
    summ = json.loads(runs[-1].read_text())
    return {t: {"valid": v["valid"], "pass_rate": v["pass_rate"]}
            for t, v in summ.get("per_task", {}).items()}


def _slug(s: str) -> str:
    """Filesystem-safe slug for a model name (for run_id suffixes)."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def _make_worker(worker_model: str) -> dict:
    """worker dict consumed by run_trial. ornith-35b routes through the in-app
    LiteLLM proxy (GPU-attributed); claude-* talks to the real Anthropic API
    (base_url None -> api.anthropic.com; key from the Modal secret)."""
    if worker_model == "ornith-35b":
        return {"model": "ornith-35b", "small_model": "ornith-35b",
                "base_url": _proxy_url(), "api_key": PROXY_MASTER_KEY}
    return {"model": worker_model,
            "small_model": "claude-haiku-4-5-20251001", "base_url": None}


def _probe_proxy(url: str) -> None:
    """Best-effort liveliness probe of the LiteLLM proxy (it serves
    /health/liveliness). Raises on non-200 so the caller can warn."""
    import urllib.request
    req = urllib.request.Request(
        f"{url}/health/liveliness",
        headers={"Authorization": f"Bearer {PROXY_MASTER_KEY}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        if r.status != 200:
            raise RuntimeError(f"proxy liveliness status {r.status}")


@app.local_entrypoint()
def run_sweep(variant: str, trials: int = 5, split: str = "dev",
              worker_model: str = "ornith-35b", tasks: str = ""):
    """Fan a variant out across the task set; write runs/<run_id>/ and append
    the cost ledger. Holdout runs require the operator's .holdout-unlocked
    flow upstream — this entrypoint refuses split='holdout' unless the flag
    file exists, as defense in depth.

    worker_model selects the coding model: 'ornith-35b' (the subject; served
    via the in-app vLLM+proxy stack, GPU-budget-gated) or a hosted 'claude-*'
    reference model (real Anthropic API spend, NOT budget-hook-gated).
    tasks: optional comma-separated task-id filter (partial run)."""
    import sys
    import time as _time
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "infra"))
    import sweep_stats as stats

    if split == "holdout" and not (root / ".holdout-unlocked").exists():
        raise SystemExit("holdout sealed: operator must create .holdout-unlocked")
    if trials < 3:
        raise SystemExit("trials < 3: paired stats need >=3 (PLAN decision rules)")

    variant_dir = root / "experiments" / "variants" / variant
    manifest = variant_dir / "manifest.yaml"
    if not manifest.exists():
        raise SystemExit(f"no such variant: {variant}")
    import yaml
    parent = yaml.safe_load(manifest.read_text()).get("parent")

    is_ornith = worker_model == "ornith-35b"
    if not is_ornith:
        # A hosted Claude worker burns real Anthropic API dollars, which the
        # budget hook does NOT gate. Require an explicit mechanical ack.
        print("WARNING: worker_model is a hosted Claude model. This spends real "
              "Anthropic API dollars the budget hook does NOT gate, and requires "
              "the 'anthropic-api-key' Modal secret to hold a real key "
              "(modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-...).")
        if os.environ.get("RUN_SWEEP_API_OK") != "1":
            raise SystemExit("refusing claude-* sweep without RUN_SWEEP_API_OK=1")

    task_dirs = sorted((root / "tasks" / split).glob("*/task.yaml"))
    if not task_dirs:
        raise SystemExit(f"no tasks in tasks/{split}/")
    specs = [_load_spec(d) for d in task_dirs]

    partial = bool(tasks.strip())
    if partial:
        wanted = {t.strip() for t in tasks.split(",") if t.strip()}
        specs = [s for s in specs if s["id"] in wanted]
        if not specs:
            raise SystemExit(f"no tasks in tasks/{split}/ match {sorted(wanted)}")

    cfg_tar = _tar_config(variant_dir)
    worker = _make_worker(worker_model)

    ts = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    run_id = f"{ts}-{variant}" if is_ornith else f"{ts}-{variant}-{_slug(worker_model)}"
    if partial:
        run_id += "-partial"

    print(f"[sweep] {run_id}: {len(specs)} tasks x {trials} trials "
          f"({len(specs) * trials} trials), variant {variant} (parent {parent}) "
          f"worker {worker_model}")

    if is_ornith:
        _wait_healthy(_serve_url())
        try:
            _probe_proxy(_proxy_url())
        except Exception as e:  # best-effort; the per-trial calls will surface it
            print(f"[sweep] proxy health probe failed (continuing): {e}")

    work = [(spec, cfg_tar, t, run_id, worker) for spec in specs for t in range(trials)]
    prov = {s["id"]: s.get("provenance") for s in specs}

    sweep_start = _time.time()
    results = []
    for r in run_trial.starmap(work):
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
            worker_model: str = "ornith-35b"):
    """One task x one trial, for exercising the runner end-to-end (the
    proof-of-one). Writes NOTHING to experiments/runs/ or the ledger — the trial
    still writes its own transcript/diff/verdict to the /runs volume — and prints
    the result dict as JSON. Refuses to reach into sealed holdout."""
    root = Path(__file__).resolve().parent.parent
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

    cfg_tar = _tar_config(variant_dir)
    worker = _make_worker(worker_model)
    if worker_model == "ornith-35b":
        _wait_healthy(_serve_url())
    ts = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    result = run_trial.remote(spec, cfg_tar, 0, f"one-{ts}-{task_id}", worker)
    print(json.dumps(result, indent=2))
