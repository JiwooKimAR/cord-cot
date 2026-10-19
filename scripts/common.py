"""Shared settings and helpers for the reproduction scripts (paper Tables 1-6).

Results layout (``--results-dir``, default ``results/``)::

    results/<model>/<task>/baseline.jsonl        01_generate_baseline.py
    results/<model>/<task>/experiment_a.jsonl    02_run_axis_a.py
    results/<model>/<task>/experiment_b.jsonl    03_run_axis_b.py
    results/ea/<model>__<task>.jsonl             04_run_early_answering.py
    results/refine/<model>/<task>/...            05-06 (self-refine, Table 4)
    results/tables/                              make_tables.py

``<model>`` is the HF id without the organisation prefix.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Paper models (HF ids), in Table 1 order.
MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/deepseek-math-7b-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
TASKS = ["gsm8k", "math500"]

DATASETS = {
    "gsm8k": {"name": "openai/gsm8k", "subset": "main", "split": "test",
              "question_field": "question", "answer_field": "answer"},
    "math500": {"name": "HuggingFaceH4/MATH-500", "subset": None, "split": "test",
                "question_field": "problem", "answer_field": "answer"},
}

# Baseline CoT budget: model override > dataset override > default.
MAX_NEW_TOKENS = 2048
MAX_NEW_TOKENS_BY_DATASET = {"math500": 4096}
MAX_NEW_TOKENS_BY_MODEL = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 16384,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 16384,
}
# vLLM context for long-CoT models (CoT 16384 + prompt + answer cue).
MAX_MODEL_LEN_BY_MODEL = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 24576,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 24576,
}


def short_name(model: str) -> str:
    return model.split("/")[-1]


def resolve_max_new_tokens(model: str, task: str) -> int:
    if model in MAX_NEW_TOKENS_BY_MODEL:
        return MAX_NEW_TOKENS_BY_MODEL[model]
    return MAX_NEW_TOKENS_BY_DATASET.get(task, MAX_NEW_TOKENS)


def engine_config(model: str, gpu_memory_utilization: float | None = None) -> dict:
    """Config dict for ``crr.models.loader`` (vLLM context = largest CoT budget + 4096)."""
    model_cfg = {"name": model, "dtype": "bfloat16"}
    if model in MAX_MODEL_LEN_BY_MODEL:
        model_cfg["max_model_len"] = MAX_MODEL_LEN_BY_MODEL[model]
    if gpu_memory_utilization is not None:
        model_cfg["gpu_memory_utilization"] = gpu_memory_utilization
    by_model = {model: MAX_NEW_TOKENS_BY_MODEL[model]} if model in MAX_NEW_TOKENS_BY_MODEL else {}
    return {
        "models": [model_cfg],
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_new_tokens_by_dataset": dict(MAX_NEW_TOKENS_BY_DATASET),
            "max_new_tokens_by_model": by_model,
        },
    }


def vllm_context_len(model: str) -> int:
    """max_model_len that ``load_vllm_engine`` uses for ``engine_config(model)``
    before clamping to the model's native limit."""
    if model in MAX_MODEL_LEN_BY_MODEL:
        return MAX_MODEL_LEN_BY_MODEL[model]
    cands = [MAX_NEW_TOKENS, *MAX_NEW_TOKENS_BY_DATASET.values()]
    if model in MAX_NEW_TOKENS_BY_MODEL:
        cands.append(MAX_NEW_TOKENS_BY_MODEL[model])
    return max(cands) + 4096


def native_max_positions(model: str) -> int | None:
    from transformers import AutoConfig
    try:
        return getattr(AutoConfig.from_pretrained(model, trust_remote_code=True),
                       "max_position_embeddings", None)
    except Exception:
        return None


def cell_dir(results_dir: str, model: str, task: str) -> str:
    path = os.path.join(results_dir, short_name(model), task)
    os.makedirs(path, exist_ok=True)
    return path


def read_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(f, rows) -> None:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()
