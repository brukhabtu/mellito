#!/usr/bin/env python3
"""Adapt our Anthropic-shaped SFT examples onto Ornith's stock qwen3_xml chat
template so that training serializes turns EXACTLY as vLLM serves them — pure
Python, stdlib + jinja2 only, so the message-shaping and template-patching logic
is unit-testable without transformers, a tokenizer, a GPU, or the network.

WHY this module exists (P4 LoRA data-prep, the train==inference invariant):

  export_trajectories.py deliberately keeps the ANTHROPIC content-block shape
  (thinking / text / tool_use / tool_result) and leaves serialization to "the
  model's own chat_template". This module is that serialization bridge. It has
  two jobs and nothing else:

  1. `anthropic_to_template_messages` — rename our block-list example into the
     OpenAI-ish dict shape the STOCK template consumes. The load-bearing detail
     is that this must match what the LiteLLM proxy hands vLLM at inference:
     reasoning goes in `reasoning_content`, tool calls in an OpenAI `tool_calls`
     list with DICT `arguments`, tool results in a flat `role: tool`
     message with `tool_call_id`. If training fed the model a different shape
     than inference, the LoRA would learn a template mismatch, not the task.

  2. `patch_template` — SFTConfig(assistant_only_loss=True) masks the loss to the
     assistant response by reading Jinja `{% generation %}...{% endgeneration %}`
     markers via the tokenizer's return_assistant_tokens_mask. Ornith's shipped
     template predates that convention and carries no markers, so we wrap the
     assistant message body in them. We cannot pin the stock template's exact
     text until runtime (it ships in the repo, trust_remote_code), so the
     insertion is a robust best-effort whose precise result is VERIFIED
     EMPIRICALLY in `preflight_template` before any GPU time is spent — a bad
     patch surfaces there for a human to hand-fix, never silently mistrains.

Only `render_and_verify` needs transformers/a tokenizer; it imports them lazily
so this module loads with just stdlib + jinja2 for the pure-function tests.
"""

import json
import re


class TemplatePatchError(Exception):
    """Raised when patch_template cannot locate the assistant branch / its
    header / its trailing <|im_end|> in the stock template. The preflight
    surfaces this so a human can hand-patch chat_template.jinja rather than
    training against a silently-wrong loss mask."""


def _join_text_blocks(blocks):
    """Concatenate the `text` field of a list of content blocks (in order).

    Used both for an assistant turn's text blocks and for a tool_result whose
    `content` is itself a list of blocks (Anthropic allows a tool_result body to
    be a string OR a list of {type:text,text:...} blocks). Non-text blocks and
    missing text are skipped, never stringified.
    """
    parts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text") or "")
    return "".join(parts)


def _tool_result_content_to_str(content):
    """Normalize a tool_result `content` to the plain string the stock template
    renders. A string passes through; a list of blocks is joined by their text
    (matching how the proxy flattens tool output before it reaches vLLM)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _join_text_blocks(content)
    if content is None:
        return ""
    # Any other JSON scalar/object: render deterministically rather than crash.
    return json.dumps(content, ensure_ascii=False)


def anthropic_to_template_messages(messages):
    """Rename our Anthropic block-list example into the stock-template shape.

    Mapping (see module docstring for WHY train==inference forces each choice):
      - system(str)  -> {"role":"system","content":<str>}
      - user(str)    -> {"role":"user","content":<str>}
      - assistant(block list) -> ONE dict:
          content          = joined `text` blocks (or "")
          reasoning_content = joined `thinking` blocks   (omitted if empty)
          tool_calls        = [{"id","type":"function",
                                "function":{"name","arguments":<input dict>}}]
                              (omitted if empty)
      - tool(block list) -> ONE dict per tool_result:
          {"role":"tool","tool_call_id":<tool_use_id>,"content":<result as str>}

    The assistant `content` key is ALWAYS present (possibly ""); only the
    reasoning_content and tool_calls keys are omitted when empty.
    """
    out = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "user"):
            # Our examples carry these as plain strings; keep them verbatim.
            out.append({"role": role, "content": content if isinstance(content, str)
                        else _tool_result_content_to_str(content)})
            continue

        if role == "assistant":
            blocks = content if isinstance(content, list) else []
            thinking_parts, text_parts, tool_calls = [], [], []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "thinking":
                    thinking_parts.append(b.get("thinking") or "")
                elif bt == "text":
                    text_parts.append(b.get("text") or "")
                elif bt == "tool_use":
                    tool_calls.append({
                        "id": b.get("id"),
                        "type": "function",
                        "function": {
                            "name": b.get("name"),
                            # Ornith's qwen3 chat template renders each tool call
                            # by iterating `arguments|items` (`<parameter=k>v</...`),
                            # so it needs a DICT, not a JSON string. Our tool_use
                            # `input` is already the arg dict — pass it through.
                            # (The template per-value does `x|string if string
                            # else x|tojson`, so nested values render correctly.)
                            "arguments": b.get("input") or {},
                        },
                    })
            asst = {"role": "assistant", "content": "".join(text_parts)}
            reasoning = "".join(thinking_parts)
            if reasoning:
                asst["reasoning_content"] = reasoning
            if tool_calls:
                asst["tool_calls"] = tool_calls
            out.append(asst)
            continue

        if role == "tool":
            blocks = content if isinstance(content, list) else []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id"),
                        "content": _tool_result_content_to_str(b.get("content")),
                    })
            continue

        # Unknown role: pass a best-effort string through rather than drop it.
        out.append({"role": role,
                    "content": content if isinstance(content, str)
                    else _tool_result_content_to_str(content)})
    return out


# Matches {% generation %} / {%- generation -%} and any whitespace-control mix.
_GENERATION_RE = re.compile(r"\{%-?\s*generation\s*-?%\}")

# Matches the assistant branch header, e.g. {%- elif message['role'] == 'assistant' %}
# (also plain `if`, single or double quotes, any whitespace-control).
_ASSISTANT_BRANCH_RE = re.compile(
    r"\{%-?\s*(?:el)?if\s+message\[\s*['\"]role['\"]\s*\]\s*==\s*"
    r"['\"]assistant['\"]\s*-?%\}"
)

# Leading keyword of any Jinja statement tag, for nesting-aware branch scanning.
_STMT_KW_RE = re.compile(r"\{%-?\s*(\w+)")


def has_generation_markers(template_str):
    """True iff the template already carries a `{% generation %}` marker (any
    whitespace-control variant). Used to short-circuit patch_template so a
    template that already supports assistant_only_loss is left untouched."""
    return bool(_GENERATION_RE.search(template_str))


def _find_branch_end(template, body_start):
    """Return the offset where the assistant branch body ends: the next
    same-nesting `elif`/`else`/`endif`/`endfor`. Inner `{% if %}` / `{% for %}`
    blocks (e.g. the tool_calls loop) increase depth so their own `endif` /
    `endfor` do NOT prematurely close the branch. Returns None if unbalanced."""
    depth = 0
    for m in _STMT_KW_RE.finditer(template, body_start):
        kw = m.group(1)
        if kw in ("if", "for"):
            depth += 1
        elif kw in ("endif", "endfor"):
            if depth == 0:
                return m.start()
            depth -= 1
        elif kw in ("elif", "else"):
            if depth == 0:
                return m.start()
    return None


def patch_template(template_str):
    """Wrap the assistant message body in `{% generation %}...{% endgeneration %}`.

    If the template already has generation markers, it is returned UNCHANGED
    (idempotent). Otherwise, best-effort:
      1. Locate the assistant branch (`elif message['role'] == 'assistant'`).
      2. Find its body end via nesting-aware scan (_find_branch_end).
      3. Insert `{% generation %}` right after the assistant header is emitted
         (just past the `}}` of the `<|im_start|>assistant` output statement).
      4. Insert `{% endgeneration %}` right before the statement that emits the
         branch's trailing `<|im_end|>`.

    The precise insertion points depend on the stock template's exact text,
    which we cannot pin until runtime; preflight_template renders real examples
    and eyeballs the resulting assistant-token mask to CONFIRM the wrap covers
    the response (thinking + text + tool_calls) and excludes the turn delimiter.
    Any structural surprise raises TemplatePatchError so it is hand-fixed, never
    silently mistrained against.
    """
    if has_generation_markers(template_str):
        return template_str

    branch = _ASSISTANT_BRANCH_RE.search(template_str)
    if not branch:
        raise TemplatePatchError(
            "no assistant branch found: expected a Jinja "
            "`{% elif message['role'] == 'assistant' %}` (or `if`) tag")

    body_start = branch.end()
    body_end = _find_branch_end(template_str, body_start)
    if body_end is None:
        raise TemplatePatchError(
            "assistant branch not terminated by a same-nesting "
            "elif/else/endif/endfor — template structure unexpected")

    body = template_str[body_start:body_end]

    # (3) Insertion point for {% generation %}: just after the output statement
    # that emits the assistant header. Find the header text, then the `}}` that
    # closes its enclosing {{ ... }} expression.
    hdr = body.find("<|im_start|>assistant")
    if hdr == -1:
        raise TemplatePatchError(
            "assistant branch does not emit '<|im_start|>assistant' — cannot "
            "locate the header to open the generation span after")
    hdr_close = body.find("}}", hdr)
    if hdr_close == -1:
        raise TemplatePatchError(
            "no `}}` closing the assistant-header output statement")
    gen_at = body_start + hdr_close + 2  # absolute offset, just past `}}`

    # (4) Insertion point for {% endgeneration %}: right before the statement
    # emitting the branch's trailing <|im_end|>. Take the LAST <|im_end|> in the
    # body and back up to the `{{` that opens its output expression.
    end_marker = body.rfind("<|im_end|>")
    if end_marker == -1:
        raise TemplatePatchError(
            "assistant branch does not emit '<|im_end|>' — cannot locate the "
            "turn delimiter to close the generation span before")
    stmt_open = body.rfind("{{", 0, end_marker)
    if stmt_open == -1:
        raise TemplatePatchError(
            "no `{{` opening the '<|im_end|>' output statement")
    endgen_at = body_start + stmt_open  # absolute offset, at the `{{`

    if not gen_at < endgen_at:
        raise TemplatePatchError(
            "assistant header and <|im_end|> are out of order — refusing to "
            "produce an inverted generation span")

    # gen_at < endgen_at, so a single splice keeps both offsets valid.
    return (template_str[:gen_at]
            + "{% generation %}"
            + template_str[gen_at:endgen_at]
            + "{% endgeneration %}"
            + template_str[endgen_at:])


def render_and_verify(tokenizer, template_str, example):
    """Render ONE example through `template_str` and report the assistant mask.

    Installs template_str as the tokenizer's chat_template, converts the
    example's messages with anthropic_to_template_messages, and tokenizes with
    return_assistant_tokens_mask so the `{% generation %}` markers yield a 0/1
    mask over the input ids. Returns a report dict:

      {"text": <decoded full render>,
       "n_tokens": <int>,
       "n_assistant_tokens": <count of masked tokens>,
       "assistant_mask_spans": [(start, end), ...],  # half-open token ranges
       "first_masked_tok": <decoded first masked token or None>,
       "last_masked_tok":  <decoded last masked token or None>}

    transformers is imported lazily (module-level pure functions must not depend
    on it); this function is exercised only in preflight, never in the CPU tests.
    """
    tokenizer.chat_template = template_str
    enc = tokenizer.apply_chat_template(
        anthropic_to_template_messages(example["messages"]),
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    input_ids = enc["input_ids"]
    # apply_chat_template may nest a single conversation in a batch dim depending
    # on the transformers version; normalize to a flat list of ids + masks.
    masks = enc.get("assistant_masks")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if masks and isinstance(masks[0], list):
        masks = masks[0]
    masks = masks or [0] * len(input_ids)

    # Contiguous runs of mask==1 -> half-open (start, end) token spans.
    spans = []
    start = None
    for i, m in enumerate(masks):
        if m and start is None:
            start = i
        elif not m and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(masks)))

    masked_idx = [i for i, m in enumerate(masks) if m]
    first_masked_tok = (tokenizer.decode([input_ids[masked_idx[0]]])
                        if masked_idx else None)
    last_masked_tok = (tokenizer.decode([input_ids[masked_idx[-1]]])
                       if masked_idx else None)

    return {
        "text": tokenizer.decode(input_ids),
        "n_tokens": len(input_ids),
        "n_assistant_tokens": len(masked_idx),
        "assistant_mask_spans": spans,
        "first_masked_tok": first_masked_tok,
        "last_masked_tok": last_masked_tok,
    }
