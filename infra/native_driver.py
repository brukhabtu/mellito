"""P9-E native minimal loop driver (mini-SWE-agent pattern) — the CONTROL arm.

Runs INSIDE the task sandbox (the same container run_trial creates for the
Claude-Code worker). It lets STOCK Ornith solve a coding task WITHOUT Claude
Code: a single session, one bash tool in a REPL loop, the model decides when it
is done. Deliberately dumb and FROZEN — this driver is never tuned. It is the
native baseline the P9 mismatch thesis is measured against, so anything that
looks like coaching, self-direction, or a completion contract is intentionally
absent (mirrors v001's minimal spirit).

Revision 2 (liveness run 20260710T203554): the original markdown-fence text
protocol was NOT the model's native transport — it degenerated into its
RL-trained qwen3_xml <tool_call> XML (django-11066: 35k chars to the token cap,
0 executed turns, ended=protocol). The vendor serves this model with the
qwen3_xml TOOL PARSER, so the transport is now the OpenAI tools API;
everything else is identical. Re-frozen after this revision.

Protocol (one turn = one bash tool call):
  - The system prompt states the task, points at VERIFY.txt, and says: use the
    bash tool; reply without calling any tool when done. Nothing else.
  - Each assistant turn is expected to carry exactly one `bash` tool_call; the
    driver runs its `command` (subprocess, cwd=/testbed, 120s/command, output
    truncated ~8k) and replies with a role:"tool" message. Multiple tool_calls
    in one turn: the FIRST is executed, the rest are dropped (recorded as a
    protocol violation in the transcript row).
  - An assistant reply with NO tool_calls means the model is done (its content
    is the final summary) — a legitimate end, never malformed.

Transport: OpenAI /v1/chat/completions directly against serve() (NOT the
LiteLLM/Anthropic proxy — that schema is Claude-Code-only). The qwen3
reasoning parser strips <think> into `reasoning_content`; we RECORD it in the
transcript but never feed it back into the conversation.

Config comes from the environment (run_trial sets these):
  NATIVE_BASE_URL   serve() web URL (we POST {base}/v1/chat/completions)
  NATIVE_API_KEY    VLLM_API_KEY (serve's Bearer gate)
  NATIVE_MODEL      served model name ("ornith-35b")
  NATIVE_TIMEOUT_S  session wall-clock budget (== the trial timeout_s)
  NATIVE_TRANSCRIPT path to write the JSONL transcript
  NATIVE_TASK       task description (or NATIVE_TASK_FILE to read it from disk)
  NATIVE_TASK_FILE  fallback file for the task description (e.g. /testbed/TASK.md)

The pure helpers (tool-call command extraction, truncation, the system prompt,
the loop against injected chat/exec callables) carry no network or subprocess
dependency, so they are unit-tested without a container.
"""

import json
import os
import subprocess
import time
import urllib.request

# --- FROZEN constants (this driver is never tuned) ------------------------
MAX_TURNS = 60              # max EXECUTED tool turns per session
PER_COMMAND_TIMEOUT = 120   # seconds per shell command
OUTPUT_LIMIT = 8000         # chars of command output returned to the model
NATIVE_MAX_TOKENS = int(os.environ.get("NATIVE_MAX_TOKENS", "12000"))
CHAT_TIMEOUT = 600          # seconds per model call (HTTP)

# The single kickoff user turn (the system prompt already carries the task).
KICKOFF = "Begin."

# The ONE tool the driver offers — a bash command in /testbed. Frozen.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in /testbed and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

# FROZEN system-prompt template. Task description is interpolated; the rest is
# fixed. Minimal, protocol only — no completion-contract / self-direction text.
_SYSTEM_TEMPLATE = """\
You are solving a software engineering task in a git repository at /testbed.

The file /testbed/VERIFY.txt contains the shell command that decides success;
run it to check your work.

Use the bash tool to run commands; when the task is complete, reply without
calling any tool.

Task:
{task}"""


def build_system_prompt(task: str) -> str:
    """The frozen system prompt with the task description interpolated."""
    return _SYSTEM_TEMPLATE.format(task=(task or "").strip())


def extract_command(tool_call) -> tuple:
    """(command, error) from one OpenAI tool_call dict.

    Valid iff the function name is `bash` and arguments carry a non-empty
    string `command` (arguments may arrive as a JSON string — the OpenAI wire
    shape — or already parsed). On any violation returns (None, reason); the
    loop reports the reason back in the tool reply and the round still counts
    as an executed turn (bounded progress; no reprompt machinery).
    """
    fn = (tool_call or {}).get("function") or {}
    name = fn.get("name")
    if name != "bash":
        return None, f"unknown tool {name!r} (only 'bash' is available)"
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return None, "arguments are not valid JSON"
    if not isinstance(args, dict):
        return None, "arguments are not an object"
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None, "missing non-empty string 'command'"
    return command, None


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
    """The frozen tool REPL. Pure w.r.t. its dependencies — `chat_fn`,
    `exec_fn`, `emit`, and `now` are all injected, so the loop is testable
    without a network or a subprocess.

      chat_fn(messages) -> {"content", "reasoning", "usage", "tool_calls"}
                           (usage is the OpenAI usage dict; may be {};
                            tool_calls a possibly-empty list).
      exec_fn(command)  -> str output (already truncated by the caller's exec).
      emit(row)         -> sink for each transcript row (dict).

    Emits, in order: the system row, the kickoff user row, then per model turn
    an assistant row (with tool_calls/reasoning/usage when present) and, when a
    tool ran, its role:"tool" reply row; finally a single
    {"type":"native_result", ...} row. Reasoning is recorded in the transcript
    but NEVER sent back into `messages`.

    Per assistant turn: no tool_calls -> the model is done (its content is the
    final summary). One or more tool_calls -> the FIRST is executed (extras
    dropped; `protocol_violation` flags the transcript row) and its output goes
    back as a role:"tool" message. The conversation history keeps ONLY the
    executed tool_call on the assistant message so every tool_call in history
    has a matching tool reply (a dangling call would break the chat template's
    multi-turn tool flow); the transcript row records ALL calls. An invalid
    call (bad args / unknown tool) gets the error text as its tool reply and
    still counts as an executed turn.

    Ends (native_result "ended"):
      done       assistant reply with no tool_calls
      max_turns  MAX_TURNS tool turns executed
      timeout    session wall-clock exceeded
      error      chat_fn raised (network/API failure)
    """
    system = build_system_prompt(task)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": kickoff}]
    emit({"role": "system", "content": system})
    emit({"role": "user", "content": kickoff})

    totals = {"tokens_in": 0, "tokens_out": 0}
    turns = 0                 # EXECUTED tool turns
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
        tool_calls = resp.get("tool_calls") or []
        totals["tokens_in"] += int(usage.get("prompt_tokens") or 0)
        totals["tokens_out"] += int(usage.get("completion_tokens") or 0)

        arow = {"role": "assistant", "content": content}
        if tool_calls:
            arow["tool_calls"] = tool_calls   # record ALL calls, even dropped ones
        if reasoning:
            arow["reasoning"] = reasoning
        if usage:
            arow["usage"] = usage
        if len(tool_calls) > 1:
            arow["protocol_violation"] = "multiple_tool_calls"
        emit(arow)

        if not tool_calls:
            # Legitimate end: the model replied without calling any tool.
            messages.append({"role": "assistant", "content": content})
            ended = "done"
            break

        call = tool_calls[0]
        command, err = extract_command(call)
        # History keeps only the EXECUTED call (see docstring). Reasoning is
        # never fed back into the conversation.
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [call]})
        if err:
            output = f"[driver] invalid tool call: {err}"
        else:
            output = exec_fn(command)
        turns += 1
        trow = {"role": "tool", "tool_call_id": (call or {}).get("id") or "",
                "content": output}
        messages.append(trow)
        emit(trow)

    emit({"type": "native_result", "turns": turns,
          "ended": ended or "unknown", "usage_total": dict(totals)})
    return {"turns": turns, "ended": ended, "totals": totals}


# --- real-transport wiring (not exercised by the unit tests) --------------

def make_chat_fn(base_url: str, api_key: str, model: str,
                 max_tokens: int = NATIVE_MAX_TOKENS):
    """OpenAI /v1/chat/completions client (urllib, no SDK dep in the sandbox)
    offering the single frozen bash tool with tool_choice auto — the model's
    native qwen3_xml transport, parsed server-side by serve()'s
    --tool-call-parser. temperature 0 mirrors the harness's determinism. Lifts
    vLLM's reasoning_content off the message when present."""
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    def chat(messages):
        body = {"model": model, "messages": messages,
                "temperature": 0, "max_tokens": max_tokens,
                "tools": [BASH_TOOL], "tool_choice": "auto"}
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
            "tool_calls": msg.get("tool_calls") or [],
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
                # universal_newlines, not text=: the driver runs under each
                # testbed's own python, and the oldest envs are 3.6 (text= is
                # a 3.7 alias; it crashed every django-11066 trial in run
                # 20260710T210023 before the first command executed).
                timeout=PER_COMMAND_TIMEOUT, universal_newlines=True,
                errors="replace")
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
