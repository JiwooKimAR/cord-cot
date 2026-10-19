"""Model and tokenizer loading utilities."""

import gc
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    config: dict,
    model_config: dict = None,
    gpu_id: int = None,
) -> tuple:
    """
    Load an HF transformers model and its (left-padded) tokenizer.

    Args:
        config: experiment config; ``config["models"][0]`` is used when
                ``model_config`` is None.
        model_config: model config dict with 'name', 'dtype' (and optionally
                      'device_map', 'attn_implementation').
        gpu_id: GPU device index to place the model on (e.g. 0 or 1).
                If None, uses device_map from model_config (default "auto").
    """
    if model_config is None:
        model_config = config.get("model", config["models"][0])

    model_name = model_config["name"]
    dtype_str = model_config.get("dtype", "bfloat16")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Determine device placement
    if gpu_id is not None:
        device_map = {"": gpu_id}
    else:
        dm = model_config.get("device_map", "auto")
        if dm == "single":
            device_map = {"": 0}
        else:
            device_map = dm

    logger.info("Loading model: %s (dtype=%s, device_map=%s)", model_name, dtype_str, device_map)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prefer SDPA (fused attention) over the eager path; it supports the
    # custom attention masks and decoding loops used by both axes.
    attn_impl = model_config.get("attn_implementation", "sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except (TypeError, ValueError) as e:
        logger.warning("attn_implementation=%s not supported (%s); falling back to default", attn_impl, e)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    model.eval()

    logger.info("Model loaded: %s",
                model.hf_device_map if hasattr(model, "hf_device_map") else "single device")

    return model, tokenizer


def load_vllm_engine(config: dict, model_config: dict = None, gpu_id: int = None):
    """
    Load a vLLM LLM engine for batch greedy generation.

    Used for plain generation (baseline CoT, A-axis answers); steps that need
    logits or custom attention masks use the HF model instead.

    ``gpu_id`` is accepted for call-site symmetry with
    ``load_model_and_tokenizer`` but does not select a device: vLLM uses the
    GPU(s) visible through ``CUDA_VISIBLE_DEVICES``.

    Returns:
        (vllm.LLM, tokenizer) tuple
    """
    from vllm import LLM

    if model_config is None:
        model_config = config.get("model", config["models"][0])

    model_name = model_config["name"]
    dtype_str = model_config.get("dtype", "bfloat16")
    gen_cfg = config.get("generation", {})
    max_new_tokens = gen_cfg.get("max_new_tokens", 2048)
    # Use the largest max_new_tokens across all datasets / models for max_model_len.
    # by_model can be keyed by either the full HF id or the short name; values for
    # the current model take precedence over dataset/global defaults.
    by_ds = gen_cfg.get("max_new_tokens_by_dataset", {}) or {}
    by_model = gen_cfg.get("max_new_tokens_by_model", {}) or {}
    candidates = [max_new_tokens] + list(by_ds.values())
    short = get_model_short_name(model_name)
    if model_name in by_model:
        candidates.append(by_model[model_name])
    if short in by_model:
        candidates.append(by_model[short])
    effective_max = max(candidates)
    # max_model_len needs to cover prompt + generation
    max_model_len = model_config.get("max_model_len", effective_max + 4096)

    # Clamp to the model's native position-embedding limit. Baseline's
    # CoT truncation ensures prompt+CoT+answer_cue fits within this limit.
    try:
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        native_limit = getattr(hf_cfg, "max_position_embeddings", None)
        if native_limit and max_model_len > native_limit:
            max_model_len = native_limit
    except Exception:
        pass

    logger.info("Loading vLLM engine: %s (dtype=%s)", model_name, dtype_str)

    gpu_mem_util = model_config.get("gpu_memory_utilization", 0.90)
    kwargs = dict(
        model=model_name,
        dtype=dtype_str,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
    )
    engine = LLM(**kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("vLLM engine loaded: %s", model_name)
    return engine, tokenizer


def unload_vllm_engine(engine):
    """Free GPU memory occupied by a vLLM engine."""
    from vllm.distributed.parallel_state import destroy_model_parallel
    # Explicitly shutdown the engine core process to release GPU memory
    engine.llm_engine.engine_core.shutdown(timeout=30)
    destroy_model_parallel()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("vLLM engine unloaded, GPU cache cleared.")


def get_model_short_name(model_name: str) -> str:
    """Extract a short name for directory/file naming."""
    short = model_name.split("/")[-1]
    return short
