#!/usr/bin/env python3
"""Convert verifier-PASSING Claude Code `stream-json` transcripts into SFT
training examples for an Ornith bf16 LoRA — pure Python, stdlib + pyyaml only,
so the whole conversion is unit-testable without Modal, a GPU, or a container.

WHY this module exists and where it sits in the pipeline (P4 data-prep):
run_sweep produces per-trial transcripts on the Modal volume; pull_transcripts
brings the passing ones local. This module is the bridge from those raw
transcripts to a training corpus. It owns exactly one decision — *what a
training example looks like* — and nothing about serving or tokenization.

Three shape facts about the transcripts drive every design choice here (all
verified against a real ornith run, not assumed):

  1. Claude Code `-p` mode emits ONE content block per JSONL line, but ALL
     consecutive `assistant` lines that share the same `message.id` are ONE
     logical assistant turn. We regroup them (group_assistant_turns) and put
     the blocks back in canonical Anthropic order: thinking, then text, then
     tool_use. Emitted order is not canonical, so we cannot rely on it.

  2. We keep the ANTHROPIC content-block shape (thinking / text / tool_use /
     tool_result), NOT an OpenAI `tool_calls` schema. The downstream trainer
     applies the MODEL'S OWN chat_template (qwen3_xml) to serialize these
     turns, so the training string matches what the model sees at inference.
     Inventing an OpenAI tool-call schema here would produce a train/inference
     mismatch. This module's job is a faithful reconstruction; the template
     owns serialization.

  3. The transcript does NOT contain Claude Code's base system prompt or the
     tool JSON schemas — line 0's `tools` field is just tool *names*. So the
     `system` message we emit is the variant's worker CLAUDE.md ONLY. That is a
     known conditioning gap between training and inference (the model was
     served under CC's full system prompt + tool schemas, which we can't
     reconstruct from the transcript); it is surfaced in build_dataset's stats
     and flagged for the operator review gate, not silently papered over.

Integrity: only verifier-PASSING trajectories are export candidates
(.claude/rules/experiment-integrity.md — "only verifier-passing trajectories
are export candidates"). build_dataset filters on verdict=="pass" from the
run's trials.jsonl; fail/invalid trials never reach the training file.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Placeholder text Claude Code emits for an assistant message whose turn is
# carried entirely by a thinking/tool_use block — it is UI filler, not a real
# assistant utterance, so it must never become a training target.
PLACEHOLDER_TEXT = "(no content)"

# len(text) / this ≈ token count. We cannot run the real tokenizer here (it is
# a heavy, GPU-side dependency and not the concern of a stdlib data-prep step),
# so token figures in the stats dict are an explicit char-based PROXY for
# order-of-magnitude budgeting only — never cite them as exact token counts.
CHARS_PER_TOKEN = 4.0


def _iter_objs(lines):
    """Yield the parsed dict for each transcript line, tolerating junk.

    Accepts a list whose items are already-parsed dicts OR raw JSON strings
    (so callers can pass a transcript file's lines directly, or inline
    fixtures). Blank lines, non-JSON partials, and non-dict values are skipped
    exactly as parse_stream_json tolerates them — a garbled line is a data
    point to survive, not a crash.
    """
    for line in lines:
        if isinstance(line, dict):
            yield line
            continue
        if not isinstance(line, str):
            continue
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue  # partial / non-JSON line — tolerate
        if isinstance(obj, dict):
            yield obj


def _clean_assistant_blocks(raw_blocks):
    """Return one assistant turn's blocks in canonical Anthropic order.

    Canonical order is thinking -> text -> tool_use (Claude Code emits them in
    arbitrary order across separate lines, so we bucket by type and concat).
    Cleaning rules:
      - thinking: keep the text, DROP `signature` — it is a placeholder from
        the non-Anthropic ornith backend, carries no training signal, and would
        just bloat the target.
      - text: drop the "(no content)" placeholder and whitespace-only text so
        empty filler never becomes a target.
      - tool_use: keep id/name/input verbatim; the id is correlated with
        tool_result later and its exact format (toolu_… vs chatcmpl-tool-…) is
        backend-specific, so we never parse or normalize it.
    """
    thinking, text, tool_use = [], [], []
    for b in raw_blocks:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "thinking":
            thinking.append({"type": "thinking", "thinking": b.get("thinking") or ""})
        elif bt == "text":
            t = b.get("text") or ""
            if t.strip() and t.strip() != PLACEHOLDER_TEXT:
                text.append({"type": "text", "text": t})
        elif bt == "tool_use":
            tool_use.append({
                "type": "tool_use",
                "id": b.get("id"),
                "name": b.get("name"),
                "input": b.get("input") or {},
            })
    return thinking + text + tool_use


def group_assistant_turns(lines):
    """Regroup a stream-json transcript into logical turns.

    Returns a flat list of turn dicts, in transcript order:
      {"role": "assistant", "id": <message.id>, "content": [<blocks>]}
      {"role": "tool",       "content": [<tool_result blocks>]}

    Grouping rules (the core, pure, testable step):
      - Consecutive `assistant` lines sharing the same `message.id` are merged
        into ONE assistant turn; blocks are re-ordered canonically by
        _clean_assistant_blocks. A new id (or any non-assistant line) flushes
        the current assistant turn.
      - Each `user` line carrying tool_result blocks becomes ONE tool turn,
        preserving tool_use_id so it stays correlated with the assistant turn's
        tool_use blocks. We keep one tool turn per user line (rather than
        merging across lines) because that is the natural 1:1 with how the
        results arrive and keeps the correlation obvious.
      - The `system` init line and the final `result` line carry no
        conversational content and are dropped here.
    """
    turns = []
    cur_id = None
    cur_blocks = []

    def flush():
        nonlocal cur_id, cur_blocks
        if cur_id is not None:
            # Emit even if content is empty after cleaning: an assistant turn
            # that only carried "(no content)" still marks a turn boundary; the
            # empty content is harmless and n_assistant_turns stays honest.
            turns.append({"role": "assistant", "id": cur_id,
                          "content": _clean_assistant_blocks(cur_blocks)})
        cur_id = None
        cur_blocks = []

    for obj in _iter_objs(lines):
        t = obj.get("type")
        if t == "assistant":
            msg = obj.get("message") or {}
            mid = msg.get("id")
            content = msg.get("content") or []
            if cur_id is not None and mid != cur_id:
                flush()
            cur_id = mid
            cur_blocks.extend(content)
        elif t == "user":
            flush()  # the preceding assistant turn is complete
            msg = obj.get("message") or {}
            results = []
            for b in (msg.get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": b.get("tool_use_id"),
                        "content": b.get("content"),
                    })
            if results:
                turns.append({"role": "tool", "content": results})
        else:
            # system / result / anything else: a turn boundary, no content.
            flush()
    flush()
    return turns


def transcript_to_messages(lines, system_text, user_text, task=None, trial=None):
    """Convert ONE trajectory into an OpenAI-style `messages` example.

    Message layout:
      [ {"role":"system",  "content": <worker CLAUDE.md text>},
        {"role":"user",    "content": <task description>},
        <assistant / tool turns reconstructed from the transcript> ]

    Assistant turns keep the Anthropic content-block LIST (thinking/text/
    tool_use); tool turns keep tool_result blocks. See the module docstring for
    why we do not convert to an OpenAI tool_calls schema.

    Returns the full example dict with light metadata used downstream for
    filtering and stats: task, trial, messages, n_assistant_turns,
    n_tool_calls, has_thinking, thinking_chars.
    """
    turns = group_assistant_turns(lines)

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    n_assistant_turns = n_tool_calls = thinking_chars = 0
    for turn in turns:
        if turn["role"] == "assistant":
            n_assistant_turns += 1
            messages.append({"role": "assistant", "content": turn["content"]})
            for b in turn["content"]:
                if b["type"] == "thinking":
                    thinking_chars += len(b["thinking"])
                elif b["type"] == "tool_use":
                    n_tool_calls += 1
        else:  # tool
            messages.append({"role": "tool", "content": turn["content"]})

    return {
        "task": task,
        "trial": trial,
        "messages": messages,
        "n_assistant_turns": n_assistant_turns,
        "n_tool_calls": n_tool_calls,
        "has_thinking": thinking_chars > 0,
        "thinking_chars": thinking_chars,
    }


# --- orchestration --------------------------------------------------------

def _local_transcript_path(run_dir, task, trial, transcript_path):
    """Map a trials.jsonl `transcript_path` to the locally-pulled file.

    The volume path is "<run_id>/<task>/trial<N>/transcript.jsonl"; pull_
    transcripts lands it at run_dir/<task>/trial<N>/transcript.jsonl where
    run_dir already ends in <run_id>. So we strip a leading <run_id>/ component
    if present and re-root under run_dir; if transcript_path is missing or
    malformed we fall back to the conventional layout.
    """
    run_dir = Path(run_dir)
    if transcript_path:
        p = Path(transcript_path)
        parts = p.parts
        if parts and parts[0] == run_dir.name:
            parts = parts[1:]  # drop the duplicated run_id component
        if parts:
            return run_dir.joinpath(*parts)
    return run_dir / task / f"trial{trial}" / "transcript.jsonl"


def _read_task_description(tasks_root, task):
    """Task prompt text from tasks/{dev,staging}/<task>/task.yaml.

    dev and staging are the two splits a passing run can draw from; holdout is
    sealed and never a training source, so it is deliberately not searched.
    """
    for split in ("dev", "staging"):
        y = Path(tasks_root) / split / task / "task.yaml"
        if y.exists():
            spec = yaml.safe_load(y.read_text()) or {}
            return spec.get("description") or ""
    return None


def _read_lines(path):
    return Path(path).read_text().splitlines()


def build_dataset(run_dir, variant_dir, tasks_root, out_path,
                  cap_per_task=3, min_thinking_chars=1):
    """Build the SFT JSONL from a run's PASSING trials and return a stats dict.

    Pipeline:
      1. Read run_dir/trials.jsonl; keep ONLY verdict=="pass" (integrity rule).
      2. For each, read its transcript, the task description, and the variant's
         worker CLAUDE.md (the system prompt), and convert.
      3. Drop trajectories with thinking_chars < min_thinking_chars — a target
         with no reasoning trace defeats the point of preserving Ornith's
         <think> reasoning (counted as dropped_no_thinking).
      4. Per task, dedupe then cap to cap_per_task (see heuristic below;
         counted as dropped_by_cap).
      5. Write one JSON example per line to out_path.

    Per-task cap heuristic (documented per spec): among a task's surviving
    trajectories we (a) drop EXACT duplicates — trajectories whose assistant/
    tool message content hashes identically, which happen when the model walks
    the same path twice — then (b) keep the SHORTEST `cap_per_task` by total
    target character count. Shorter passing trajectories tend to be the ones
    without dead-ends or backtracking, giving cleaner, cheaper SFT targets;
    dedupe-first keeps the kept set diverse rather than N copies of one path.
    """
    run_dir = Path(run_dir)
    variant_dir = Path(variant_dir)
    system_text = (variant_dir / "claude-config" / "CLAUDE.md").read_text()

    trials = list(_iter_objs(_read_lines(run_dir / "trials.jsonl")))
    passing = [t for t in trials if t.get("verdict") == "pass"]

    dropped_no_thinking = 0
    dropped_missing = 0
    by_task = {}  # task -> list of (target_len, content_hash, example)

    for t in passing:
        task = t.get("task")
        trial = t.get("trial")
        tpath = _local_transcript_path(run_dir, task, trial,
                                       t.get("transcript_path"))
        if not tpath.exists():
            dropped_missing += 1
            continue
        user_text = _read_task_description(tasks_root, task)
        if user_text is None:
            dropped_missing += 1
            continue
        ex = transcript_to_messages(_read_lines(tpath), system_text, user_text,
                                    task=task, trial=trial)
        if ex["thinking_chars"] < min_thinking_chars:
            dropped_no_thinking += 1
            continue
        # Serialize only the model-produced turns (assistant + tool) for the
        # dedupe key and the length metric; system/user are identical across a
        # task's trials, so including them would mask real duplicates.
        target = ex["messages"][2:]
        blob = json.dumps(target, sort_keys=True, ensure_ascii=False)
        by_task.setdefault(task, []).append(
            (len(blob), hashlib.sha1(blob.encode()).hexdigest(), ex))

    dropped_by_cap = 0
    kept = []
    per_task_counts = {}
    for task, cands in by_task.items():
        seen = set()
        deduped = []
        for length, h, ex in cands:
            if h in seen:
                dropped_by_cap += 1
                continue
            seen.add(h)
            deduped.append((length, ex))
        deduped.sort(key=lambda x: x[0])  # shortest first
        keep = deduped[:cap_per_task]
        dropped_by_cap += len(deduped) - len(keep)
        per_task_counts[task] = len(keep)
        kept.extend(ex for _, ex in keep)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Char-based token PROXY, broken down by role (see CHARS_PER_TOKEN note).
    tok_by_role = {}
    for ex in kept:
        for m in ex["messages"]:
            c = m["content"]
            chars = len(c) if isinstance(c, str) else len(
                json.dumps(c, ensure_ascii=False))
            tok_by_role[m["role"]] = tok_by_role.get(m["role"], 0) + chars
    token_estimate = {r: int(v / CHARS_PER_TOKEN) for r, v in tok_by_role.items()}
    token_estimate["total"] = sum(token_estimate.values())

    return {
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
        "out_path": str(out_path),
        "n_passing_trials": len(passing),
        "n_examples": len(kept),
        "n_tasks_covered": len(per_task_counts),
        "per_task_counts": per_task_counts,
        "dropped_no_thinking": dropped_no_thinking,
        "dropped_by_cap": dropped_by_cap,
        "dropped_missing": dropped_missing,
        "token_estimate_proxy": token_estimate,
        # Conditioning gap flagged for the operator review gate: the `system`
        # target is the variant CLAUDE.md only. Claude Code's base system prompt
        # and tool JSON schemas were present at serving time but are NOT in the
        # transcript, so training conditions on strictly less than inference.
        "conditioning_note": ("system prompt = variant CLAUDE.md only; CC base "
                              "system prompt + tool schemas are absent from "
                              "transcripts (train/inference gap)"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="experiments/runs/<run_id> (already pulled)")
    ap.add_argument("--variant", required=True,
                    help="variant name, e.g. v002-completion-contract")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--cap", type=int, default=3,
                    help="max trajectories kept per task (default 3)")
    ap.add_argument("--tasks-root", default=str(ROOT / "tasks"))
    ap.add_argument("--variants-root",
                    default=str(ROOT / "experiments" / "variants"))
    ap.add_argument("--min-thinking-chars", type=int, default=1)
    args = ap.parse_args()

    variant_dir = Path(args.variants_root) / args.variant
    if not variant_dir.exists():
        sys.exit(f"variant dir not found: {variant_dir}")

    stats = build_dataset(args.run_dir, variant_dir, args.tasks_root, args.out,
                          cap_per_task=args.cap,
                          min_thinking_chars=args.min_thinking_chars)

    print("SFT export complete")
    print(f"  out:            {stats['out_path']}")
    print(f"  passing trials: {stats['n_passing_trials']}")
    print(f"  examples:       {stats['n_examples']} "
          f"across {stats['n_tasks_covered']} tasks")
    print(f"  dropped:        no_thinking={stats['dropped_no_thinking']} "
          f"by_cap={stats['dropped_by_cap']} missing={stats['dropped_missing']}")
    te = stats["token_estimate_proxy"]
    print(f"  token proxy:    total~{te['total']} "
          f"(by role: {{" + ", ".join(f"{r}:{te[r]}" for r in te if r != 'total')
          + "}})")
    print(f"  per task:       {stats['per_task_counts']}")
    print(f"  NOTE: {stats['conditioning_note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
