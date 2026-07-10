"""Unit tests for the PURE helpers in infra/train_pref.py — the class-balance
weight and the length-stat percentiles are stdlib-only and run anywhere. The
preference loss itself needs torch, so it is guarded by importorskip: it is
skipped where torch is absent (e.g. the CPU harness) and its real proof is the
synthetic forward/backward in preflight_pref on Modal. Importing train_pref pulls
in `modal` (present in the harness) but no torch/trl/peft.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import train_pref as tp


# (a) neg_weight_from_counts: neg_lambda * n_pos/n_neg, guarded at n_neg==0. -----

def test_neg_weight_balances_the_count_imbalance():
    # The documented default corpus (~96 pos, ~173 neg) at neg_lambda=0.2.
    w = tp.neg_weight_from_counts(96, 173, 0.2)
    assert w == pytest.approx(0.2 * 96 / 173)
    # Summed over an epoch the negative mass is neg_lambda times the positive:
    #   Σ_neg = n_neg * w * mean(UL) == neg_lambda * n_pos * mean(UL)
    assert 173 * w == pytest.approx(0.2 * 96)


def test_neg_weight_scales_linearly_with_lambda():
    assert tp.neg_weight_from_counts(100, 100, 0.5) == pytest.approx(0.5)
    assert tp.neg_weight_from_counts(100, 100, 0.0) == pytest.approx(0.0)


def test_neg_weight_zero_when_no_negatives():
    # No fail rows -> weight 0 -> degenerates to pure SFT on the positives.
    assert tp.neg_weight_from_counts(96, 0, 0.2) == 0.0


# (b) _length_stats: nearest-rank percentiles + over-max count. ------------------

def test_length_stats_basic():
    st = tp._length_stats([10, 20, 30, 40, 100], max_length=50)
    assert st["count"] == 5
    assert st["max"] == 100
    assert st["over_max_length"] == 1          # only 100 exceeds 50
    assert st["p50"] in (20, 30)               # nearest-rank median
    assert st["p90"] in (40, 100)


def test_length_stats_empty():
    st = tp._length_stats([], max_length=32768)
    assert st == {"count": 0, "p50": 0, "p90": 0, "max": 0, "over_max_length": 0}


def test_length_stats_all_over():
    st = tp._length_stats([40000, 50000], max_length=32768)
    assert st["over_max_length"] == 2


# (c) pref_loss_from_logits: pos=CE, neg=bounded UL, assistant-only gradient. -----
# torch-only; skipped where torch is unavailable.

def test_pref_loss_pos_is_token_cross_entropy():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    b, t, v = 1, 4, 5
    logits = torch.randn(b, t, v)
    labels = torch.full((b, t), -100, dtype=torch.long)
    labels[0, 1:] = torch.tensor([1, 2, 3])     # assistant tokens
    is_pos = torch.tensor([1.0])                 # positive row

    loss = tp.pref_loss_from_logits(logits, labels, is_pos, neg_weight=0.11)

    # Reference: mean token CE over the assistant positions.
    shift_logits = logits[:, :-1, :].float().reshape(-1, v)
    shift_labels = labels[:, 1:].reshape(-1)
    ref = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)


def test_pref_loss_negative_uses_bounded_unlikelihood():
    torch = pytest.importorskip("torch")

    b, t, v = 1, 4, 5
    logits = torch.randn(b, t, v)
    labels = torch.full((b, t), -100, dtype=torch.long)
    labels[0, 1:] = torch.tensor([1, 2, 3])
    is_pos = torch.tensor([0.0])                 # negative row

    loss = tp.pref_loss_from_logits(logits, labels, is_pos, neg_weight=0.11)
    # Unlikelihood −log(1−p) is >= 0 and finite (bounded, not the runaway −λ·CE).
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_pref_loss_negative_does_not_explode_when_confident():
    torch = pytest.importorskip("torch")

    b, t, v = 1, 3, 4
    # Make the model near-certain of the true token (p -> 1) — the −log(1−p)
    # danger zone. The clamp must keep the loss finite.
    logits = torch.zeros(b, t, v)
    logits[0, 0, 2] = 50.0                        # position 0 predicts token idx 2
    labels = torch.tensor([[-100, 2, -100]])      # assistant token at pos 1
    is_pos = torch.tensor([0.0])

    loss = tp.pref_loss_from_logits(logits, labels, is_pos, neg_weight=1.0,
                                    neg_p_clamp=1e-4)
    assert torch.isfinite(loss)
    # With p clamped at 1-1e-4, −log(1−p) == −log(1e-4) ≈ 9.21.
    assert loss.item() == pytest.approx(-torch.log(torch.tensor(1e-4)).item(),
                                        rel=1e-3)


def test_pref_loss_gradient_is_zero_on_nonassistant_tokens():
    torch = pytest.importorskip("torch")

    b, t, v = 2, 6, 8
    logits = torch.randn(b, t, v, requires_grad=True)
    labels = torch.full((b, t), -100, dtype=torch.long)
    labels[0, 3:] = torch.randint(0, v, (t - 3,))
    labels[1, 3:] = torch.randint(0, v, (t - 3,))
    is_pos = torch.tensor([1.0, 0.0])            # one pass, one fail

    loss = tp.pref_loss_from_logits(logits, labels, is_pos, neg_weight=0.11)
    loss.backward()

    # logits[:, i, :] predicts token i+1, so it is masked iff labels[:, i+1]==-100.
    shift_valid = labels[:, 1:].ne(-100)
    for i in range(b):
        for pos in range(t - 1):
            if not bool(shift_valid[i, pos]):
                assert int(torch.count_nonzero(logits.grad[i, pos])) == 0, (
                    f"gradient leaked onto non-assistant position ({i},{pos})")
    # The last (sliced-off) position never receives gradient.
    assert int(torch.count_nonzero(logits.grad[:, t - 1])) == 0


def test_pref_loss_class_weight_scales_negative_term():
    torch = pytest.importorskip("torch")

    b, t, v = 1, 4, 5
    logits = torch.randn(b, t, v)
    labels = torch.full((b, t), -100, dtype=torch.long)
    labels[0, 1:] = torch.tensor([1, 2, 3])
    is_pos = torch.tensor([0.0])                 # negative row

    l1 = tp.pref_loss_from_logits(logits, labels, is_pos, neg_weight=0.10)
    l2 = tp.pref_loss_from_logits(logits, labels, is_pos, neg_weight=0.20)
    # Negative loss is linear in neg_weight -> doubling the weight doubles it.
    assert l2.item() == pytest.approx(2.0 * l1.item(), rel=1e-5)
