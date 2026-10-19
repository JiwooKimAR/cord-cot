"""Inverse decoding (A-axis): replace the CoT suffix with rank-selected tokens.

Structure: X -> [0-p% original CoT] -> [p-100% inverse-token replacement] -> answer cue -> Y_A

Token selection (``rank_position``):
  - ``top-k`` / ``bottom-k``: the exact k-th most / least probable valid token
  - ``random``: uniform over valid tokens;  ``kl``: sampled from Q(t) proportional to 1/p(t)

Valid tokens exclude special, non-Latin-script and thinking-marker tokens;
``set_special_tokens(tokenizer)`` must be called once before selection.

Two-pass use:
  Pass 1 (HF): one forward pass over prompt + baseline CoT gives the logits at
      every position; inverse tokens for all (rank x point) conditions are
      selected from them (a single ``torch.topk`` over the (N, V) slice) and
      the edited prompts are built.
  Pass 2 (vLLM or HF): the answer is generated for each edited prompt.
"""

import unicodedata

import torch

from crr.prompts import ANSWER_CUE

# Token IDs that should never be selected during inverse decoding
_SPECIAL_TOKEN_IDS = set()


def _is_excluded_piece(piece):
    """Return True if this decoded token piece should never be sampled.

    Used to keep inverse decoding from collapsing generation into noise:
    empty/whitespace-only pieces, control characters, surrogates / private
    use codepoints, and non-Latin alphabetic scripts (CJK/Arabic/Cyrillic/
    Hebrew/etc.) are all excluded. ASCII + Latin-1 letters, digits, math
    symbols and standard punctuation remain selectable - bottom-rank
    perturbations should still be able to pick up unusual English tokens
    and LaTeX, just not wander into other scripts or garbage.
    """
    if piece is None:
        return True
    s = piece
    # SentencePiece / GPT-2 space markers - treat as whitespace for this check.
    s = s.replace('▁', ' ').replace('Ġ', ' ').replace('Ċ', '\n')
    if s.strip() == '':
        return True
    for ch in s:
        cat = unicodedata.category(ch)
        # Control / surrogate / private use / unassigned / format.
        if cat in ('Cc', 'Cs', 'Co', 'Cn', 'Cf'):
            return True
        if ord(ch) < 32 or ord(ch) == 127:
            return True
        # Letters outside Latin script. Digits, math symbols, and ASCII
        # punctuation all carry non-letter categories, so this only filters
        # genuine foreign-script alphabet tokens.
        if cat.startswith('L'):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                return True
            if not name.startswith('LATIN'):
                return True
    return False


_THINKING_PIECES = (
    "<think>", "</think>",
    "<｜begin▁of▁thinking｜>", "<｜end▁of▁thinking｜>",
    "<|begin_of_thinking|>", "<|end_of_thinking|>",
)


def set_special_tokens(tokenizer):
    """Register token IDs to exclude from inverse selection.

    Excludes (1) tokenizer-declared special ids (eos/bos/pad/unk/etc.),
    (2) vocab entries whose decoded piece is empty, a control character,
    or belongs to a non-Latin script, and (3) reasoning-model thinking
    markers (`<think>`, `</think>`, etc.) that are ASCII and would
    otherwise pass the script filter. The thinking exclusion is what
    keeps DeepSeek-R1-Distill's bottom-rank perturbations from injecting
    `</think>` into the middle of a CoT and short-circuiting reasoning.
    """
    global _SPECIAL_TOKEN_IDS
    special = set()
    if tokenizer.eos_token_id is not None:
        special.add(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        special.add(tokenizer.bos_token_id)
    if tokenizer.pad_token_id is not None:
        special.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        special.add(tokenizer.unk_token_id)
    for tok_id in tokenizer.all_special_ids:
        special.add(tok_id)

    # (3) Hardcoded thinking markers across known reasoning models.
    for piece in _THINKING_PIECES:
        try:
            tid = tokenizer.convert_tokens_to_ids(piece)
        except Exception:
            tid = None
        if tid is None:
            continue
        unk = tokenizer.unk_token_id
        if tid == unk:
            continue
        if isinstance(tid, int) and tid >= 0:
            special.add(tid)

    # Forward compatibility: any added token whose surface form mentions
    # "think" gets excluded. Catches future variants without code changes.
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tid, tok in added.items():
        try:
            content = tok.content if hasattr(tok, "content") else str(tok)
        except Exception:
            continue
        if "think" in content.lower():
            try:
                special.add(int(tid))
            except (TypeError, ValueError):
                pass

    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    try:
        pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    except Exception:
        pieces = None
    if pieces is not None:
        for tid, piece in enumerate(pieces):
            if tid in special:
                continue
            if _is_excluded_piece(piece):
                special.add(tid)

    _SPECIAL_TOKEN_IDS = special


def _get_valid_mask(probs: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask where True = valid (non-special) token."""
    if not _SPECIAL_TOKEN_IDS:
        raise RuntimeError(
            "No excluded token ids registered - call "
            "crr.axes.necessity.set_special_tokens(tokenizer) before selecting "
            "inverse tokens."
        )
    mask = torch.ones(probs.shape[-1], dtype=torch.bool, device=probs.device)
    for tid in _SPECIAL_TOKEN_IDS:
        if tid < probs.shape[-1]:
            mask[tid] = False
    return mask


@torch.no_grad()
def _select_inverse_tokens_vectorized(
    logits: torch.Tensor,
    n_tokens: int,
    start_pos: int,
    rank_position: str,
    temperature: float = 1.0,
    rng_seed: int = None,
) -> list:
    """Select inverse tokens from pre-computed logits using vectorized ops.

    Instead of looping over positions and calling argsort per position,
    slices the logits into (n_tokens, vocab) and uses a single topk call.

    Args:
        logits: (1, seq_len, vocab_size) - full sequence logits.
        n_tokens: number of tokens to select
        start_pos: position in logits where replacement starts.
        rank_position: "bottom-1", "bottom-5", "bottom-10", "top-5", "top-10",
            "random", "kl"
        temperature: softmax temperature
        rng_seed: when provided, used to construct a per-call torch.Generator
            so that the stochastic branches ("random", "kl") are deterministic
            across resumes/reruns. Deterministic branches ignore it.

    Returns:
        list of selected token IDs
    """
    end_pos = min(start_pos + n_tokens, logits.shape[1])
    actual_n = end_pos - start_pos
    if actual_n <= 0:
        return []

    # Slice: (actual_n, vocab)
    chunk = logits[0, start_pos:end_pos, :]
    probs = torch.softmax(chunk / temperature, dim=-1)

    # Valid mask: same across all positions (vocab-level)
    valid_mask = _get_valid_mask(probs[0])  # (vocab,)

    # Per-call generator for stochastic branches. Same device as probs so
    # that torch.{randint,multinomial} accept it.
    gen = None
    if rng_seed is not None and rank_position in ("random", "kl"):
        gen = torch.Generator(device=probs.device)
        gen.manual_seed(int(rng_seed))

    if rank_position == "random":
        # Uniform random from valid tokens, per position
        valid_indices = torch.where(valid_mask)[0]  # (n_valid,)
        n_valid = valid_indices.shape[0]
        picks = torch.randint(
            0, n_valid, (actual_n,), device=probs.device, generator=gen,
        )
        selected = valid_indices[picks]
        return selected.tolist()

    if rank_position == "kl":
        # KL-divergence inverse sampling: sample from Q(t) proportional to 1/(p_base + eps),
        # restricted to valid tokens. A third perturbation family distinct
        # from rank-based (top/bottom-k, deterministic) and uniform-random
        # (flat over V_valid). Tokens with low p_base contribute most to
        # D_KL(delta_t || p_base) = -log p_base(t), so Q concentrates mass on
        # low-prob tokens without being a hard argmin. Used to check that
        # results are not an artifact of the bottom-k selection rule.
        eps = 1e-8
        inv_weight = torch.zeros_like(probs)
        inv_weight[:, valid_mask] = 1.0 / (probs[:, valid_mask] + eps)
        inv_weight = inv_weight / inv_weight.sum(dim=-1, keepdim=True)
        selected = torch.multinomial(
            inv_weight, num_samples=1, generator=gen,
        ).squeeze(-1)  # (actual_n,)
        return selected.tolist()

    # Mask invalid tokens: set to -inf for top-k, +inf for bottom-k
    # so topk ignores them.
    is_bottom = rank_position.startswith("bottom-")
    is_top = rank_position.startswith("top-")

    if is_bottom:
        k = int(rank_position.split("-")[1])
        # We want the k-th smallest among valid tokens.
        # Set invalid to +inf so they don't appear in smallest-k.
        masked_probs = probs.masked_fill(~valid_mask.unsqueeze(0), float("inf"))
        # topk with largest=False gives k smallest values
        _vals, indices_in_vocab = torch.topk(masked_probs, k, dim=-1, largest=False)
        # Take the k-th (last) one - that's the k-th from bottom
        selected = indices_in_vocab[:, -1]  # (actual_n,)
        return selected.tolist()

    elif is_top:
        k = int(rank_position.split("-")[1])
        # Set invalid to -inf so they don't appear in largest-k.
        masked_probs = probs.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
        _vals, indices_in_vocab = torch.topk(masked_probs, k, dim=-1, largest=True)
        # Take the k-th (last) one - that's the k-th from top
        selected = indices_in_vocab[:, -1]  # (actual_n,)
        return selected.tolist()

    else:
        raise ValueError(f"Unsupported rank_position for vectorized selection: {rank_position}")


@torch.no_grad()
def compute_baseline_logits(
    model,
    prompt_input_ids: torch.LongTensor,
    baseline_ids: list,
) -> tuple:
    """Compute logits for prompt + baseline CoT in a single forward pass.

    Returns:
        (logits, prompt_len) where logits is (1, T_total, vocab_size).
    """
    device = prompt_input_ids.device
    cot_tensor = torch.tensor([baseline_ids], device=device, dtype=torch.long)
    full_input = torch.cat([prompt_input_ids, cot_tensor], dim=1)
    full_mask = torch.ones_like(full_input)
    outputs = model(input_ids=full_input, attention_mask=full_mask)
    return outputs.logits, prompt_input_ids.shape[1]


def select_inverse_tokens(
    cached_logits: torch.Tensor,
    prompt_len: int,
    baseline_ids: list,
    rank_position: str,
    intervention_point: int,
    temperature: float = 1.0,
    rng_seed: int = None,
) -> dict:
    """Select inverse tokens for a single condition using cached logits.

    ``cached_logits`` come from an unpadded forward (``compute_baseline_logits``);
    this is ``select_inverse_tokens_batched_view`` with no left padding.

    Returns:
        dict with replacement_ids, replacement_start, n_replaced,
        n_cot_original, edited_cot_ids.
    """
    return select_inverse_tokens_batched_view(
        cached_logits, 0, prompt_len, baseline_ids, rank_position,
        intervention_point, temperature=temperature, rng_seed=rng_seed,
    )


@torch.no_grad()
def compute_baseline_logits_batched(
    model,
    tokenizer,
    prompts: list,
    baseline_ids_list: list,
) -> tuple:
    """Batched single-forward over (prompt + baseline_cot) for B instances.

    Rows are left-padded here with ``tokenizer.pad_token_id``.

    Returns:
        logits: (B, T_pad, V)
        attention_mask: (B, T_pad)
        prompt_lens: list[int] of length B - non-padded prompt token count per row
        pad_left_counts: list[int] of length B - left-pad amount per row
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    prompt_ids_list = [tokenizer.encode(p) for p in prompts]
    full_ids_list = [pi + bi for pi, bi in zip(prompt_ids_list, baseline_ids_list)]
    T_pad = max(len(x) for x in full_ids_list)

    B = len(full_ids_list)
    input_ids = torch.full((B, T_pad), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T_pad), dtype=torch.long, device=device)
    position_ids = torch.zeros((B, T_pad), dtype=torch.long, device=device)
    pad_left_counts = []
    for i, full_ids in enumerate(full_ids_list):
        pl = T_pad - len(full_ids)
        pad_left_counts.append(pl)
        input_ids[i, pl:] = torch.tensor(full_ids, dtype=torch.long, device=device)
        attention_mask[i, pl:] = 1
        # Explicit position_ids so absolute-position-embedding models (e.g. GPT-2)
        # behave identically under left-padding. RoPE models ignore this gracefully.
        position_ids[i, pl:] = torch.arange(len(full_ids), dtype=torch.long, device=device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    prompt_lens = [len(pi) for pi in prompt_ids_list]
    return outputs.logits, attention_mask, prompt_lens, pad_left_counts


def select_inverse_tokens_batched_view(
    row_logits: torch.Tensor,
    pad_left_count: int,
    prompt_len: int,
    baseline_ids: list,
    rank_position: str,
    intervention_point: int,
    temperature: float = 1.0,
    rng_seed: int = None,
    filler_token_ids: list = None,
) -> dict:
    """Per-row adapter for batched logits.

    `row_logits` is a (1, T_pad, V) slice from the batched tensor (e.g.
    `cached_logits_BTV[i:i+1]`). Position arithmetic adds `pad_left_count`
    to account for left-padding; ``select_inverse_tokens`` is this function
    with ``pad_left_count == 0``.

    `rng_seed` (used only by stochastic ranks: "random", "kl") makes
    selection deterministic across resumes - pass a per-(query, condition)
    seed so that re-running yields identical edited prompts.

    Special baseline cases of `rank_position` (the in-distribution
    baselines of Lanham et al. 2023):
      - "early_answering": only *removes* the last (1-p)% of the CoT (no
        replacement). Lanham's best-diagnosticity baseline.
      - "filler": fills the last (1-p)% of the CoT with ellipsis tokens
        (in-distribution). Maps to Lanham's filler tokens.
        `filler_token_ids` must provide the ellipsis token id list
        (usually 1-3 tokens), cycled to length n_replace.

    Returns same dict schema as `select_inverse_tokens`.
    """
    n_cot = len(baseline_ids)
    replacement_start = max(0, min(n_cot, int(n_cot * intervention_point / 100)))
    n_replace = n_cot - replacement_start

    if rank_position == "early_answering":
        # Remove CoT[replacement_start:] (truncation only, no replacement).
        # Empty replacement leaves only the prefix in edited_cot_ids.
        replacement_ids = []
    elif rank_position == "filler":
        # Fill n_replace positions with ellipsis tokens, cycling
        # filler_token_ids if it is shorter.
        if not filler_token_ids:
            raise ValueError("rank_position='filler' requires filler_token_ids")
        if n_replace > 0:
            replacement_ids = [
                filler_token_ids[i % len(filler_token_ids)]
                for i in range(n_replace)
            ]
        else:
            replacement_ids = []
    elif n_replace > 0:
        logits_start = pad_left_count + prompt_len + replacement_start - 1
        if logits_start < 0:
            logits_start = 0
        replacement_ids = _select_inverse_tokens_vectorized(
            row_logits, n_replace, logits_start, rank_position, temperature,
            rng_seed=rng_seed,
        )
    else:
        replacement_ids = []

    edited_cot_ids = list(baseline_ids[:replacement_start]) + list(replacement_ids)
    return {
        "replacement_ids": replacement_ids,
        "replacement_start": replacement_start,
        "n_replaced": len(replacement_ids),
        "n_cot_original": n_cot,
        "edited_cot_ids": edited_cot_ids,
    }


def build_edited_prompt(
    tokenizer,
    prompt_text: str,
    edited_cot_ids: list,
    max_total_tokens: int = None,
) -> str:
    """Build the full edited prompt string for answer generation.

    If max_total_tokens is set, truncates the edited CoT so that the
    total tokenized length (prompt + cot + answer_cue) fits within the
    limit. This prevents vLLM from rejecting prompts that exceed the
    model's max_position_embeddings after decode/re-encode drift.

    Returns:
        prompt_text + decoded(edited_cot) + answer_cue
    """
    answer_cue = ANSWER_CUE
    edited_cot_text = tokenizer.decode(edited_cot_ids, skip_special_tokens=True)
    full = prompt_text + edited_cot_text + answer_cue

    if max_total_tokens is not None:
        ids = tokenizer.encode(full)
        if len(ids) > max_total_tokens:
            # Truncate CoT text by re-encoding and trimming tokens
            prompt_ids = tokenizer.encode(prompt_text)
            cue_ids = tokenizer.encode(answer_cue, add_special_tokens=False)
            max_cot_tokens = max_total_tokens - len(prompt_ids) - len(cue_ids)
            if max_cot_tokens > 0:
                cot_ids = tokenizer.encode(edited_cot_text, add_special_tokens=False)
                cot_ids = cot_ids[:max_cot_tokens]
                edited_cot_text = tokenizer.decode(cot_ids, skip_special_tokens=True)
                full = prompt_text + edited_cot_text + answer_cue

    return full
