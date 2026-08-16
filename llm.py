"""
Ollama wrapper. No Streamlit imports. Three functions only.
"""

import time
import json
import subprocess
from collections.abc import Iterator

import ollama


def health_check() -> tuple[bool, str, list[str]]:
    """
    Returns (ok, message, local_model_tags).
    Pings the server and lists locally available models.
    """
    try:
        models = ollama.list()
        tags = [m.model for m in models.models]
        return True, f"Ollama running — {len(tags)} model(s) available", tags
    except Exception as e:
        return False, f"Ollama unreachable: {e}\n\nFix: run  ollama serve", []


def chat(
    model: str,
    messages: list[dict],
    num_predict: int,
    think: bool = False,
    temperature: float = 0.8,
) -> tuple[str, float]:
    """
    Blocking call. Returns (text, wall_seconds).
    Used for judge calls and closing statements.
    """
    t0 = time.perf_counter()
    # think is a top-level ollama.chat param, not inside options.
    # gemma4/qwen3.x default thinking ON; content field stays empty during
    # the thinking phase. Pass think=False to get content directly.
    resp = ollama.chat(
        model=model,
        messages=messages,
        think=think,
        options={"num_predict": num_predict, "temperature": temperature},
    )
    elapsed = time.perf_counter() - t0
    text = resp.message.content or ""
    return text, elapsed


def chat_stream(
    model: str,
    messages: list[dict],
    num_predict: int,
    temperature: float = 0.9,
    think: bool = False,
) -> Iterator[str]:
    """
    Yields token deltas. Used for fighter turns.
    think=False by default — gemma4 and qwen3.x put ALL output in
    chunk.message.thinking when thinking mode is on; content stays empty.

    After the generator is exhausted, callers can read production metrics:
        chat_stream._last_duration   — wall seconds
        chat_stream._last_ttft       — seconds to first token
        chat_stream._last_tokens_out — output token count (eval_count)
        chat_stream._last_tokens_in  — prompt token count (prompt_eval_count)
        chat_stream._last_tps        — tokens/second (eval_count / eval_duration)
    """
    t0 = time.perf_counter()
    first_token_t: float = 0.0

    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        think=think,
        options={"num_predict": num_predict, "temperature": temperature},
    )

    for chunk in stream:
        delta = chunk.message.content or ""
        if delta:
            if not first_token_t:
                first_token_t = time.perf_counter()
            yield delta
        if chunk.done:
            t_end = time.perf_counter()
            tokens_out = getattr(chunk, "eval_count", 0) or 0
            eval_dur_ns = getattr(chunk, "eval_duration", 0) or 0
            chat_stream._last_duration = t_end - t0                           # type: ignore[attr-defined]
            chat_stream._last_ttft = (first_token_t - t0) if first_token_t else 0.0  # type: ignore
            chat_stream._last_tokens_out = tokens_out                         # type: ignore
            chat_stream._last_tokens_in = getattr(chunk, "prompt_eval_count", 0) or 0  # type: ignore
            chat_stream._last_tps = (tokens_out / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0  # type: ignore
            return

    # Fallback: done chunk never fired
    t_end = time.perf_counter()
    chat_stream._last_duration = t_end - t0              # type: ignore
    chat_stream._last_ttft = (first_token_t - t0) if first_token_t else 0.0  # type: ignore
    chat_stream._last_tokens_out = 0                     # type: ignore
    chat_stream._last_tokens_in = 0                      # type: ignore
    chat_stream._last_tps = 0.0                          # type: ignore


def warmup(models: list[str]) -> dict[str, float]:
    """
    Sends a 1-token request to each model in order.
    Returns {tag: seconds} timing per model.
    """
    timings: dict[str, float] = {}
    for tag in models:
        t0 = time.perf_counter()
        try:
            ollama.chat(
                model=tag,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 1},
            )
        except Exception as e:
            timings[tag] = -1.0
            print(f"  [warmup] {tag} FAILED: {e}")
            continue
        elapsed = time.perf_counter() - t0
        timings[tag] = elapsed
        print(f"  [warmup] {tag}: {elapsed:.1f}s")
    return timings


# ---------------------------------------------------------------------------
# CP1 smoke test — run with: python llm.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== CP1: Ollama health check ===")
    ok, msg, tags = health_check()
    print(f"Status: {msg}")
    if tags:
        print("Models available:")
        for t in tags:
            print(f"  {t}")
    if not ok:
        print("\nStart Ollama with:  ollama serve")
        sys.exit(1)

    # Pick the first available model for a quick stream test
    if not tags:
        print("No models pulled. Pull one with: ollama pull mistral:7b")
        sys.exit(1)

    test_model = tags[0]
    print(f"\n=== Streaming test on {test_model!r} ===")
    messages = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "In exactly two sentences, what is a debate?"},
    ]

    print("Response: ", end="", flush=True)
    gen = chat_stream(test_model, messages, num_predict=60)
    collected = ""
    for token in gen:
        print(token, end="", flush=True)
        collected += token
    duration = getattr(chat_stream, "_last_duration", 0.0)
    print(f"\nDuration: {duration:.2f}s  |  Chars: {len(collected)}  |  Words: {len(collected.split())}")

    print(f"\n=== Blocking call test on {test_model!r} ===")
    text, secs = chat(test_model, messages, num_predict=60, temperature=0.5)
    print(f"Response: {text.strip()}")
    print(f"Duration: {secs:.2f}s")

    print("\nCP1 PASSED")
