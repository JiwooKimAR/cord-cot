"""CPU unit tests for the measurement layer (no model, no GPU).

Uses synthetic logits and a minimal fake tokenizer. Skipped when torch is not
installed (``pip install -e ".[measure]"``).
"""

import pytest

torch = pytest.importorskip("torch")

import crr.axes.necessity as nec  # noqa: E402
from crr.axes.sufficiency import _visible_cut  # noqa: E402
from crr.models.chat_template import render_chat  # noqa: E402
from crr.prompts import ANSWER_CUE  # noqa: E402

V = 12            # vocab size
EXCLUDED = {0, 11}  # pretend special / non-Latin ids
PROMPT_LEN = 3
BASE_IDS = list(range(100, 120))  # 20 CoT tokens (ids outside the fake vocab are fine)


@pytest.fixture
def excluded(monkeypatch):
    monkeypatch.setattr(nec, "_SPECIAL_TOKEN_IDS", set(EXCLUDED))


def _logits(T=PROMPT_LEN + len(BASE_IDS)):
    # probability increases with token id at every position
    return torch.arange(V, dtype=torch.float32).repeat(1, T, 1)


# ---------------------------------------------------------------------------
# rank selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rank, expected", [
    ("bottom-1", 1), ("bottom-5", 5), ("bottom-10", 10),
    ("top-1", 10), ("top-5", 6), ("top-10", 1),
])
def test_rank_selection_skips_excluded_ids(excluded, rank, expected):
    out = nec.select_inverse_tokens(_logits(), PROMPT_LEN, BASE_IDS, rank, 0)
    assert set(out["replacement_ids"]) == {expected}


def test_selection_requires_set_special_tokens(monkeypatch):
    monkeypatch.setattr(nec, "_SPECIAL_TOKEN_IDS", set())
    with pytest.raises(RuntimeError, match="set_special_tokens"):
        nec.select_inverse_tokens(_logits(), PROMPT_LEN, BASE_IDS, "bottom-1", 50)


@pytest.mark.parametrize("p, start", [(0, 0), (25, 5), (50, 10), (95, 19), (100, 20)])
def test_replacement_window(excluded, p, start):
    out = nec.select_inverse_tokens(_logits(), PROMPT_LEN, BASE_IDS, "bottom-1", p)
    assert out["replacement_start"] == start
    assert out["n_replaced"] == len(BASE_IDS) - start
    assert out["n_cot_original"] == len(BASE_IDS)
    assert out["edited_cot_ids"][:start] == BASE_IDS[:start]
    assert len(out["edited_cot_ids"]) == len(BASE_IDS)


def test_batched_view_matches_unpadded(excluded):
    pad = 4
    logits = _logits()
    padded = torch.cat([torch.zeros(1, pad, V), logits], dim=1)
    for rank in ("bottom-1", "top-5"):
        for p in (0, 35, 100):
            a = nec.select_inverse_tokens(logits, PROMPT_LEN, BASE_IDS, rank, p)
            b = nec.select_inverse_tokens_batched_view(padded, pad, PROMPT_LEN, BASE_IDS, rank, p)
            assert a == b


def test_random_rank_seeded_and_valid(excluded):
    a = nec.select_inverse_tokens(_logits(), PROMPT_LEN, BASE_IDS, "random", 0, rng_seed=7)
    b = nec.select_inverse_tokens(_logits(), PROMPT_LEN, BASE_IDS, "random", 0, rng_seed=7)
    assert a == b
    assert not set(a["replacement_ids"]) & EXCLUDED


def test_early_answering_and_filler(excluded):
    ea = nec.select_inverse_tokens_batched_view(_logits(), 0, PROMPT_LEN, BASE_IDS,
                                                "early_answering", 50)
    assert ea["edited_cot_ids"] == BASE_IDS[:10]
    fl = nec.select_inverse_tokens_batched_view(_logits(), 0, PROMPT_LEN, BASE_IDS,
                                                "filler", 50, filler_token_ids=[7, 8])
    assert fl["edited_cot_ids"] == BASE_IDS[:10] + [7, 8] * 5


def test_unknown_rank(excluded):
    with pytest.raises(ValueError):
        nec.select_inverse_tokens(_logits(), PROMPT_LEN, BASE_IDS, "middle-3", 0)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    bos_token = "<s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return self.bos_token + body + ("[assistant]" if add_generation_prompt else "")

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def test_render_chat_strips_leading_bos():
    out = render_chat(_FakeTokenizer(), [{"role": "user", "content": "q"}])
    assert out == "[user]q[assistant]"


def test_build_edited_prompt_appends_answer_cue():
    out = nec.build_edited_prompt(_FakeTokenizer(), "PROMPT|", [1, 2, 3])
    assert out == "PROMPT|1 2 3" + ANSWER_CUE


# ---------------------------------------------------------------------------
# B-axis visible cut (token unit)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p, n_visible", [(0, 0), (33, 13), (50, 20), (100, 40)])
def test_visible_cut_token(p, n_visible):
    n, meta = _visible_cut(None, "", [], 40, p, "token")
    assert n == n_visible
    assert meta["n_masked_tokens"] == 40 - n_visible
    assert meta["mask_unit"] == "token"


def test_visible_cut_unknown_unit():
    with pytest.raises(ValueError):
        _visible_cut(None, "", [], 40, 50, "word")
