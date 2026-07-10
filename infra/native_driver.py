"""P9-E native minimal loop driver (mini-SWE-agent pattern) — the CONTROL arm.

Runs INSIDE the task sandbox (the same container run_trial creates for the
Claude-Code worker). It lets STOCK Ornith solve a coding task WITHOUT Claude
Code: a single session, a pure-text shell-REPL loop, the model decides when it
is done. Deliberately dumb and FROZEN — this driver is never tuned. It is the
native baseline the P9 mismatch thesis is measured against, so anything that
looks like coaching, self-direction, or a completion contract is intentionally
absent (mirrors v001's minimal spirit).

Protocol (one turn = one shell command):
  - The system prompt states the task, points at VERIFY.txt, and gives the
    protocol rules — nothing else.
  - Each assistant turn contains exactly one ```bash fenced command; the driver
    runs it (subprocess, cwd=/testbed, 120s/command, output truncated ~8k) and
    returns the output as the next user message.
  - The model finishes by emitting DONE on a line by itself instead of a command.

Transport: OpenAI /v1/chat/completions directly against serve() (NOT the
LiteLLM/Anthropic proxy — that schema is Claude-Code-only). NO tools param;
pure text is the most native protocol for Ornith's RL. The qwen3 reasoning
parser strips <think> into `reasoning_content`; we RECORD it in the transcript
but never feed it back into the conversation.

Config comes from the environment (run_trial sets these):
  NATIVE_BASE_URL   serve() web URL (we POST {base}/v1/chat/completions)
  NATIVE_API_KEY    VLLM_API_KEY (serve's Bearer gate)
  NATIVE_MODEL      served model name ("ornith-35b")
  NATIVE_TIMEOUT_S  session wall-clock budget (== the trial timeout_s)
  NATIVE_TRANSCRIPT path to write the JSONL transcript
  NATIVE_TASK       task description (or NATIVE_TASK_FILE to read it from disk)
  NATIVE_TASK_FILE  fallback file for the task description (e.g. /testbed/TASK.md)

The pure helpers (extract/classify a turn, truncate output, build the system
prompt, run the loop against injected chat/exec callables) carry no network or
subprocess dependency, so they are unit-tested without a container.
"""

import json
import os
import re
import subprocess
import time
import urllib.request

# --- FROZEN constants (this driver is never tuned) ------------------------
MAX_TURNS = 60              # max EXECUTED shell commands per session
PER_COMMAND_TIMEOUT = 120   # seconds per shell command
OUTPUT_LIMIT = 8000         # chars of command output returned to the model
NATIVE_MAX_TOKENS = int(os.environ.get("NATIVE_MAX_TOKENS", "6000"))
CHAT_TIMEOUT = 600          # seconds per model call (HTTP)

# The single kickoff user turn (the system prompt already carries the task).
# Protocol-level only — no task-specific coaching.
KICKOFF = "Begin."

# One-line reminder used exactly once after a malformed turn (see run_loop).
PROTOCOL_REMINDER = (
    "Reply with exactly one ```bash command block, or DONE on a line by "
    "itself to finish."
)

# FROZEN system-prompt template. Task description is interpolated; the rest is
# fixed. <20 lines, protocol only — no completion-contract / self-direction text.
_SYSTEM_TEMPLATE = """\
You are solving a software engineering task in a git repository at /testbed.

The file /testbed/VERIFY.txt contains the shell command that decides success;
run it to check your work.

Protocol (follow it exactly):
- Reply with your reasoning, then exactly ONE shell command to run, inside a
  single ```bash fenced code block.
- I will execute that command in /testbed and reply with its output.
- Send one command per turn and wait for its output before sending the next.
- When the task is complete, reply with DONE on a line by itself and no command.

Task:
{task}"""


def build_system_prompt(task: str) -> str:
    """The frozen system prompt with the task description interpolated."""
    return _SYSTEM_TEMPLATE.format(task=(task or "").strip())


# A recognized shell fence: ```bash / ```sh / ```shell then a newline, then the
# command body up to the closing fence. A bare ``` fence (no shell language) is
# NOT a command — that keeps a markdown code block of prose from being executed,
# and an off-protocol turn is honestly scored as malformed.
_FENCE_RE = re.compile(
    r"```[ \t]*(?:bash|sh|shell)[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_bash_blocks(text: str) -> list:
    """Every ```bash|sh|shell fenced command body in `text`, in order."""
    return [m.group(1).strip() for m in _FENCE_RE.finditer(text or "")]


def _has_done_line(text: str) -> bool:
    """True iff some line of `text` is exactly DONE (stripped)."""
    return any(line.strip() == "DONE" for line in (text or "").splitlines())


def classify_turn(text: str) -> dict:
    """Classify one assistant turn's content into the protocol's three cases.

    Returns {"kind", "command", "multiple_blocks"}:
      - a shell fence present -> kind="command", command=the FIRST block's body;
        multiple_blocks=True flags a protocol violation (we still run the first
        only). A command present ALWAYS wins over a stray DONE line — an emitted
        command means the model is not finished.
      - no command but a lone DONE line -> kind="done".
      - neither -> kind="malformed".
    """
    blocks = extract_bash_blocks(text)
    if blocks:
        return {"kind": "command", "command": blocks[0],
                "multiple_blocks": len(blocks) > 1}
    if _has_done_line(text):
        return {"kind": "done", "command": None, "multiple_blocks": False}
    return {"kind": "malformed", "command": None, "multiple_blocks": False}


def truncate_output(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Bound command output to ~`limit` chars, keeping head AND tail (errors
    usually surface at the end) with an explicit truncation marker."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n...[{omitted} chars truncated]...\n{text[-tail:]}"


def run_loop(chat_fn, exec_fn, task, *, emit, timeout_s=None,
             max_turns=MAX_TURNS, now=time.time, kickoff=KICKOFF):
    """The frozen text REPL. Pure w.r.t. its dependencies — `chat_fn`,
    `exec_fn`, `emit`, and `now` are all injected, so the loop is testable
    without a network or a subprocess.

      chat_fn(messages) -> {"content", "reasoning", "usage"} (usage is the
                           OpenAI usage dict; may be {}).
      exec_fn(command)  -> str output (already truncated by the caller's exec).
      emit(row)         -> sink for each transcript row (dict).

    Emits, in order: the system row, the kickoff user row, then per model turn
    an assistant row (+ reasoning/usage when present) and either a command's
    output user row or a one-time protocol-reminder user row, and finally a
    single {"type":"native_result", ...} row. Reasoning is recorded in the
    transcript but NEVER sent back into `messages`.

    Ends (native_result "ended"):
      done       model emitted DONE
      max_turns  MAX_TURNS commands executed
      timeout    session wall-clock exceeded
      protocol   two consecutive malformed turns
      error      chat_fn raised (network/API failure)
    """
    system = build_system_prompt(task)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": kickoff}]
    emit({"role": "system", "content": system})
    emit({"role": "user", "content": kickoff})

    totals = {"tokens_in": 0, "tokens_out": 0}
    turns = 0                 # EXECUTED command turns (malformed reprompts don't count)
    consecutive_malformed = 0
    ended = None
    start = now()

    while True:
        if turns >= max_turns:
            ended = "max_turns"
            break
        if timeout_s is not None and (now() - start) >= timeout_s:
            ended = "timeout"
            break

        try:
            resp = chat_fn(messages)
        except Exception as e:  # network/API failure — end gracefully
            ended = "error"
            emit({"role": "system", "content": f"[driver] chat error: {e}"})
            break

        content = resp.get("content") or ""
        reasoning = resp.get("reasoning") or ""
        usage = resp.get("usage") or {}
        totals["tokens_in"] += int(usage.get("prompt_tokens") or 0)
        totals["tokens_out"] += int(usage.get("completion_tokens") or 0)

        info = classify_turn(content)
        arow = {"role": "assistant", "content": content}
        if reasoning:
            arow["reasoning"] = reasoning
        if usage:
            arow["usage"] = usage
        if info["multiple_blocks"]:
            arow["protocol_violation"] = "multiple_bash_blocks"
        emit(arow)
        # Never feed reasoning_content back into the conversation.
        messages.append({"role": "assistant", "content": content})

        kind = info["kind"]
        if kind == "done":
            ended = "done"
            break
        if kind == "malformed":
            consecutive_malformed += 1
            if consecutive_malformed >= 2:
                ended = "protocol"
                break
            messages.append({"role": "user", "content": PROTOCOL_REMINDER})
            emit({"role": "user", "content": PROTOCOL_REMINDER})
            continue

        # kind == "command"
        consecutive_malformed = 0
        output = exec_fn(info["command"])
        turns += 1
        messages.append({"role": "user", "content": output})
        emit({"role": "user", "content": output})

    emit({"type": "native_result", "turns": turns,
          "ended": ended or "unknown", "usage_total": dict(totals)})
    return {"turns": turns, "ended": ended, "totals": totals}


# --- real-transport wiring (not exercised by the unit tests) --------------

def make_chat_fn(base_url: str, api_key: str, model: str,
                 max_tokens: int = NATIVE_MAX_TOKENS):
    """OpenAI /v1/chat/completions client (urllib, no SDK dep in the sandbox).
    NO tools param — pure text. temperature 0 mirrors the harness's determinism.
    Lifts vLLM's reasoning_content off the message when present."""
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    def chat(messages):
        body = {"model": model, "messages": messages,
                "temperature": 0, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            endpoint, data=json.dumps(body).encode(), headers=headers,
            method="POST")
        with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT) as r:
            resp = json.loads(r.read())
        msg = resp["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "usage": resp.get("usage") or {},
        }

    return chat


def make_exec_fn():
    """Run one shell command in /testbed (stderr folded into stdout), bounded by
    PER_COMMAND_TIMEOUT, output truncated to OUTPUT_LIMIT. Never raises — a
    timeout or nonzero exit is reported as text, the loop keeps going."""
    def run(command):
        try:
            p = subprocess.run(
                ["bash", "-lc", command], cwd="/testbed",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=PER_COMMAND_TIMEOUT, text=True, errors="replace")
            out = (p.stdout or "") + f"\n[exit code: {p.returncode}]"
        except subprocess.TimeoutExpired as e:
            partial = e.output or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            out = partial + f"\n[command timed out after {PER_COMMAND_TIMEOUT}s]"
        return truncate_output(out)

    return run


def _read_task() -> str:
    t = os.environ.get("NATIVE_TASK")
    if t:
        return t
    f = os.environ.get("NATIVE_TASK_FILE")
    if f and os.path.exists(f):
        with open(f, encoding="utf-8", errors="replace") as fp:
            return fp.read()
    return ""


def main():
    base_url = os.environ["NATIVE_BASE_URL"]
    api_key = os.environ.get("NATIVE_API_KEY", "")
    model = os.environ.get("NATIVE_MODEL", "ornith-35b")
    timeout_s = float(os.environ.get("NATIVE_TIMEOUT_S", "1800"))
    transcript = os.environ.get("NATIVE_TRANSCRIPT", "/tmp/native_transcript.jsonl")
    task = _read_task()

    chat_fn = make_chat_fn(base_url, api_key, model)
    exec_fn = make_exec_fn()

    fh = open(transcript, "w", encoding="utf-8")

    def emit(row):
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    try:
        run_loop(chat_fn, exec_fn, task, emit=emit, timeout_s=timeout_s)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
