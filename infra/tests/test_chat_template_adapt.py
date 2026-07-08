"""Unit tests for infra/chat_template_adapt.py — the Anthropic-block ->
stock-template message renaming and the `{% generation %}` template patch. Pure
functions over inline fixtures, so no transformers, tokenizer, GPU, or network
is required. render_and_verify is deliberately NOT tested here — it needs a real
tokenizer and its correctness is confirmed empirically in preflight_template.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from chat_template_adapt import (
    anthropic_to_template_messages,
    has_generation_markers,
    patch_template,
    TemplatePatchError,
)


# (a) anthropic_to_template_messages: block-list -> stock-template dicts. -----

def test_assistant_maps_thinking_text_and_tool_use():
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "let me reason"},
            {"type": "text", "text": "I'll read a file."},
            {"type": "tool_use", "id": "chatcmpl-tool-1", "name": "Read",
             "input": {"file_path": "/x.py"}},
        ]},
        {"role": "tool", "content": [
            {"type": "tool_result", "tool_use_id": "chatcmpl-tool-1",
             "content": "file body"},
        ]},
    ]
    out = anthropic_to_template_messages(messages)

    assert out[0] == {"role": "system", "content": "sys prompt"}
    assert out[1] == {"role": "user", "content": "do the thing"}

    asst = out[2]
    assert asst["role"] == "assistant"
    # thinking -> reasoning_content, text -> content
    assert asst["reasoning_content"] == "let me reason"
    assert asst["content"] == "I'll read a file."
    # tool_use -> OpenAI tool_calls with JSON-STRING arguments
    assert asst["tool_calls"] == [{
        "id": "chatcmpl-tool-1", "type": "function",
        "function": {"name": "Read",
                     "arguments": json.dumps({"file_path": "/x.py"})},
    }]
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == \
        {"file_path": "/x.py"}

    # tool_result -> flat role:tool with tool_call_id
    tool = out[3]
    assert tool == {"role": "tool", "tool_call_id": "chatcmpl-tool-1",
                    "content": "file body"}


def test_assistant_joins_multiple_blocks_in_order():
    messages = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "part1 "},
        {"type": "thinking", "thinking": "part2"},
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]}]
    out = anthropic_to_template_messages(messages)
    assert out[0]["reasoning_content"] == "part1 part2"
    assert out[0]["content"] == "hello world"
    assert "tool_calls" not in out[0]


def test_empty_reasoning_and_tool_calls_keys_omitted():
    # Assistant carrying only text: content present, other keys absent.
    out = anthropic_to_template_messages(
        [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}])
    assert out[0] == {"role": "assistant", "content": "hi"}
    assert "reasoning_content" not in out[0]
    assert "tool_calls" not in out[0]

    # Assistant carrying only a tool_use: content is "" (always present).
    out = anthropic_to_template_messages(
        [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": "ls"}}]}])
    assert out[0]["content"] == ""
    assert "reasoning_content" not in out[0]
    assert out[0]["tool_calls"][0]["function"]["name"] == "Bash"


def test_tool_result_list_content_joined_to_text():
    # A tool_result whose content is a list of blocks -> joined text string.
    out = anthropic_to_template_messages([{"role": "tool", "content": [
        {"type": "tool_result", "tool_use_id": "t9", "content": [
            {"type": "text", "text": "line1\n"},
            {"type": "text", "text": "line2"},
        ]},
    ]}])
    assert out[0] == {"role": "tool", "tool_call_id": "t9",
                      "content": "line1\nline2"}


def test_multiple_tool_results_become_multiple_tool_messages():
    out = anthropic_to_template_messages([{"role": "tool", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "ra"},
        {"type": "tool_result", "tool_use_id": "b", "content": "rb"},
    ]}])
    assert out == [
        {"role": "tool", "tool_call_id": "a", "content": "ra"},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
    ]


# (b) has_generation_markers: true/false across whitespace-control variants. --

def test_has_generation_markers_true_variants():
    assert has_generation_markers("x {% generation %} y")
    assert has_generation_markers("x {%- generation -%} y")
    assert has_generation_markers("x {%-generation-%} y")
    assert has_generation_markers("x {%  generation  %} y")


def test_has_generation_markers_false():
    assert not has_generation_markers("{%- if x %}{{ y }}{%- endif %}")
    assert not has_generation_markers("plain text with the word generation")


# A small qwen3-like template with an `elif role == 'assistant'` branch whose
# body contains its OWN inner {% if %}/{% endif %} (the nesting the patcher must
# see through) and a trailing <|im_end|>.
QWEN_LIKE = (
    "{%- for message in messages %}\n"
    "{%- if message['role'] == 'system' %}\n"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{%- elif message['role'] == 'user' %}\n"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{%- elif message['role'] == 'assistant' %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- if message['content'] %}\n"
    "{{- message['content'] }}\n"
    "{%- endif %}\n"
    "{{- '<|im_end|>\\n' }}\n"
    "{%- endif %}\n"
    "{%- endfor %}"
)


# (c) patch_template inserts markers around the assistant body and is idempotent.

def test_patch_template_inserts_markers_around_assistant_body():
    patched = patch_template(QWEN_LIKE)
    assert has_generation_markers(patched)
    assert "{% generation %}" in patched
    assert "{% endgeneration %}" in patched

    gi = patched.index("{% generation %}")
    ei = patched.index("{% endgeneration %}")
    # generation opens after the assistant header, endgeneration closes before it.
    assert gi < ei
    # The header emission is BEFORE the generation open (header not in mask body
    # start... it is emitted, then generation opens right after).
    hdr = patched.index("<|im_start|>assistant")
    assert hdr < gi
    # The inner content statement sits INSIDE the generation span.
    inner = patched.index("message['content'] }}")
    assert gi < inner < ei
    # The trailing <|im_end|> output sits AFTER endgeneration (excluded).
    last_imend_stmt = patched.rfind("'<|im_end|>\\n' }}")
    assert ei < last_imend_stmt


def test_patch_template_idempotent_when_already_marked():
    once = patch_template(QWEN_LIKE)
    twice = patch_template(once)
    # Already-marked -> returned unchanged (single generation marker, no nesting).
    assert twice == once
    assert once.count("{% generation %}") == 1
    assert once.count("{% endgeneration %}") == 1


def test_patch_template_preserves_already_marked_template_verbatim():
    already = "{%- elif message['role'] == 'assistant' %}{% generation %}body{% endgeneration %}<|im_end|>"
    assert patch_template(already) == already


# (d) patch_template raises on a template with no assistant branch. -----------

def test_patch_template_raises_without_assistant_branch():
    no_asst = (
        "{%- for message in messages %}\n"
        "{%- if message['role'] == 'user' %}\n"
        "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}\n"
        "{%- endif %}\n"
        "{%- endfor %}"
    )
    with pytest.raises(TemplatePatchError):
        patch_template(no_asst)
