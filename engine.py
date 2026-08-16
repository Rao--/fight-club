"""
Debate orchestration. Pure logic — no Streamlit imports.
"""

import json
import random
import warnings
from typing import Callable, Iterator, Optional

import config
import llm
import prompts
from schemas import BlockVerdict, Turn, TurnScore, fallback_verdict

# ---------------------------------------------------------------------------
# State shape expected from callers (mirrors session_state keys)
# ---------------------------------------------------------------------------
# {
#   "topic": str,
#   "fighter_a": {"model": ModelSpec, "persona": Persona},
#   "fighter_b": {"model": ModelSpec, "persona": Persona},
#   "judge_model": str,
#   "exchange_num": int,
#   "turns": list[Turn],
#   "verdicts": list[BlockVerdict],
#   "timings": list[dict],
#   "speaks_first": str,           # "A" or "B", randomised per exchange
# }


def _build_fighter_messages(
    state: dict,
    fighter_letter: str,
    exchange_num: int,
) -> list[dict]:
    cfg = state["fighter_a"] if fighter_letter == "A" else state["fighter_b"]
    persona = cfg["persona"]
    model_spec = cfg["model"]
    position = "FOR" if fighter_letter == "A" else "AGAINST"

    # From exchange 3 onwards, inject aggression if a fighter is behind
    aggressive = False
    if exchange_num >= 3:
        a_total, b_total = cumulative_scores(state["verdicts"])
        if fighter_letter == "A" and a_total < b_total:
            aggressive = True
        elif fighter_letter == "B" and b_total < a_total:
            aggressive = True

    system = prompts.fighter_system(
        persona_name=persona.name,
        persona_system_prompt=persona.system_prompt,
        fighter_letter=fighter_letter,
        topic=state["topic"],
        position=position,
    )
    user = prompts.fighter_user(
        exchange_num=exchange_num,
        turns=[t for t in state["turns"] if not t.is_closing],
        persona_name=persona.name,
        fighter_letter=fighter_letter,
        aggressive=aggressive,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_exchange(
    state: dict,
    stream_callback: Optional[Callable[[str, str], Iterator[str]]] = None,
) -> tuple[Turn, Turn, Optional[BlockVerdict]]:
    """
    Runs one complete exchange: first speaker, second speaker, optional judge.
    stream_callback(fighter_letter, model_tag) → yields token strings; caller
    is responsible for displaying them. If None, turns are generated blocking.

    Returns (turn_first, turn_second, verdict_or_None).
    Mutates state["turns"], state["verdicts"], state["timings"] in place.
    """
    exchange_num = state["exchange_num"]
    speaks_first = state.get("speaks_first", "A")
    speaks_second = "B" if speaks_first == "A" else "A"

    results: dict[str, Turn] = {}

    for fighter_letter in [speaks_first, speaks_second]:
        cfg = state["fighter_a"] if fighter_letter == "A" else state["fighter_b"]
        model_tag = cfg["model"].tag
        persona = cfg["persona"]

        messages = _build_fighter_messages(state, fighter_letter, exchange_num)

        if stream_callback is not None:
            # Streaming path — caller collects text
            gen = stream_callback(fighter_letter, model_tag)
            content = "".join(gen)
            elapsed = getattr(llm.chat_stream, "_last_duration", 0.0)
        else:
            content, elapsed = llm.chat(
                model=model_tag,
                messages=messages,
                num_predict=config.FIGHTER_NUM_PREDICT,
                temperature=config.FIGHTER_TEMPERATURE,
                think=False,
            )

        if elapsed > config.FIGHTER_CEILING_S:
            warnings.warn(
                f"Fighter {fighter_letter} ({model_tag}) took {elapsed:.1f}s "
                f"(ceiling {config.FIGHTER_CEILING_S}s)"
            )

        turn = Turn(
            exchange_num=exchange_num,
            fighter=fighter_letter,
            model=model_tag,
            persona_name=persona.name,
            content=content.strip(),
            seconds=elapsed,
        )
        state["turns"].append(turn)
        state["timings"].append({"kind": "fighter", "model": model_tag, "seconds": elapsed})
        results[fighter_letter] = turn

    # Judge if this exchange triggers a block
    verdict: Optional[BlockVerdict] = None
    if exchange_num in config.JUDGE_AFTER:
        verdict = _run_judge(state)
        if verdict:
            state["verdicts"].append(verdict)

    # Randomise first speaker for the next exchange
    state["speaks_first"] = random.choice(["A", "B"])

    return results[speaks_first], results[speaks_second], verdict


def _run_judge(state: dict) -> Optional[BlockVerdict]:
    """
    Calls the judge for the current block; handles retry + fallback.
    """
    exchange_num = state["exchange_num"]
    block_num = len(state["verdicts"]) + 1

    # Determine which exchanges this block covers
    judged_so_far: list[int] = []
    for v in state["verdicts"]:
        judged_so_far.extend(v.covers_exchanges)
    block_exchanges = [e for e in range(1, exchange_num + 1) if e not in judged_so_far]
    all_exchanges_so_far = list(range(1, exchange_num + 1))

    system_msg = prompts.judge_system()
    user_msg = prompts.judge_user(
        topic=state["topic"],
        all_turns=state["turns"],
        block_num=block_num,
        block_exchanges=block_exchanges,
        all_exchanges_so_far=all_exchanges_so_far,
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    t_start = 0.0
    import time
    t_start = time.perf_counter()

    raw, elapsed = llm.chat(
        model=state["judge_model"],
        messages=messages,
        num_predict=config.JUDGE_NUM_PREDICT,
        think=config.JUDGE_THINK,
        temperature=config.JUDGE_TEMPERATURE,
    )

    if elapsed > config.JUDGE_CEILING_S:
        warnings.warn(
            f"Judge ({state['judge_model']}) took {elapsed:.1f}s "
            f"(ceiling {config.JUDGE_CEILING_S}s)"
        )

    state["timings"].append({"kind": "judge", "model": state["judge_model"], "seconds": elapsed})

    verdict = _parse_judge_output(raw, block_num, block_exchanges)
    if verdict is not None:
        return verdict

    # Retry once with correction note appended
    messages_retry = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": "Your previous response was not valid JSON. Respond with ONLY the JSON object."},
    ]
    raw2, elapsed2 = llm.chat(
        model=state["judge_model"],
        messages=messages_retry,
        num_predict=config.JUDGE_NUM_PREDICT,
        think=config.JUDGE_THINK,
        temperature=config.JUDGE_TEMPERATURE,
    )
    state["timings"].append({"kind": "judge_retry", "model": state["judge_model"], "seconds": elapsed2})

    verdict2 = _parse_judge_output(raw2, block_num, block_exchanges)
    if verdict2 is not None:
        return verdict2

    # Both attempts failed — return neutral fallback
    print(f"[engine] Judge failed twice on block {block_num}. Using fallback verdict.")
    return fallback_verdict(block_num, block_exchanges)


def _parse_judge_output(raw: str, block_num: int, block_exchanges: list[int]) -> Optional[BlockVerdict]:
    """
    Strip fences, extract JSON, validate. Returns None on failure.
    """
    text = raw.strip()
    # Strip markdown fences
    for fence in ["```json", "```"]:
        text = text.replace(fence, "")
    # Extract substring from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    json_str = text[start : end + 1]
    try:
        return BlockVerdict.model_validate_json(json_str)
    except Exception:
        return None


def run_closings(
    state: dict,
    stream_callback: Optional[Callable[[str, str], Iterator[str]]] = None,
) -> tuple[Turn, Turn]:
    """
    Generates closing statements for both fighters (blind — each without seeing the other's).
    Returns (closing_a, closing_b). Appends to state["turns"].
    """
    debate_turns = [t for t in state["turns"] if not t.is_closing]
    closings: dict[str, Turn] = {}

    for fighter_letter in ["A", "B"]:
        cfg = state["fighter_a"] if fighter_letter == "A" else state["fighter_b"]
        model_tag = cfg["model"].tag
        persona = cfg["persona"]
        position = "FOR" if fighter_letter == "A" else "AGAINST"

        system = prompts.fighter_system(
            persona_name=persona.name,
            persona_system_prompt=persona.system_prompt,
            fighter_letter=fighter_letter,
            topic=state["topic"],
            position=position,
        )
        user = prompts.closing_user(
            topic=state["topic"],
            turns=debate_turns,
            persona_name=persona.name,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if stream_callback is not None:
            gen = stream_callback(fighter_letter, model_tag)
            content = "".join(gen)
            elapsed = getattr(llm.chat_stream, "_last_duration", 0.0)
        else:
            content, elapsed = llm.chat(
                model=model_tag,
                messages=messages,
                num_predict=config.CLOSING_NUM_PREDICT,
                temperature=config.FIGHTER_TEMPERATURE,
                think=False,
            )

        if elapsed > config.CLOSING_CEILING_S:
            warnings.warn(
                f"Closing {fighter_letter} ({model_tag}) took {elapsed:.1f}s "
                f"(ceiling {config.CLOSING_CEILING_S}s)"
            )

        turn = Turn(
            exchange_num=0,
            fighter=fighter_letter,
            model=model_tag,
            persona_name=persona.name,
            content=content.strip(),
            seconds=elapsed,
            is_closing=True,
        )
        state["turns"].append(turn)
        state["timings"].append({"kind": "closing", "model": model_tag, "seconds": elapsed})
        closings[fighter_letter] = turn

    return closings["A"], closings["B"]


def cumulative_scores(verdicts: list[BlockVerdict]) -> tuple[int, int]:
    """Returns (A_total, B_total) summed raw scores across all judged blocks."""
    a_total = 0
    b_total = 0
    for v in verdicts:
        for score in v.scores:
            if score.fighter == "A":
                a_total += score.total
            else:
                b_total += score.total
    return a_total, b_total


# ---------------------------------------------------------------------------
# CP2 / CP3 smoke test — run with: python engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    print("=== CP2: Headless debate ===")

    # Check Ollama
    ok, msg, tags = llm.health_check()
    print(f"Ollama: {msg}")
    if not ok:
        sys.exit(1)

    # Hard-code default models for headless test
    a_tag = config.DEFAULT_FIGHTER_A_TAG
    b_tag = config.DEFAULT_FIGHTER_B_TAG
    judge_tag = config.JUDGE_MODEL

    missing = [t for t in [a_tag, b_tag, judge_tag] if t not in tags]
    if missing:
        print(f"\nMissing models: {missing}")
        for m in missing:
            print(f"  ollama pull {m}")
        sys.exit(1)

    persona_a = config.PERSONAS[0]  # Cynical VC
    persona_b = config.PERSONAS[1]  # Idealistic Grad

    state = {
        "topic": "This house believes that remote work has made us worse at our jobs.",
        "fighter_a": {
            "model": config.MODELS_BY_TAG[a_tag],
            "persona": persona_a,
        },
        "fighter_b": {
            "model": config.MODELS_BY_TAG[b_tag],
            "persona": persona_b,
        },
        "judge_model": judge_tag,
        "exchange_num": 1,
        "turns": [],
        "verdicts": [],
        "timings": [],
        "speaks_first": "A",
    }

    print(f"\nFighter A: {a_tag} as {persona_a.name}")
    print(f"Fighter B: {b_tag} as {persona_b.name}")
    print(f"Judge:     {judge_tag}")
    print(f"Topic:     {state['topic']}\n")
    print("-" * 60)

    debate_start = time.perf_counter()

    for exchange_num in range(1, config.MAX_EXCHANGES + 1):
        state["exchange_num"] = exchange_num
        print(f"\n--- EXCHANGE {exchange_num} ---")

        t1, t2, verdict = run_exchange(state)

        for turn in [t1, t2]:
            wc = len(turn.content.split())
            print(f"\nFIGHTER {turn.fighter} ({turn.persona_name}) [{wc} words, {turn.seconds:.1f}s]:")
            print(turn.content)

        if verdict:
            print(f"\n[BLOCK {verdict.block_num} — {verdict.block_winner.upper()} wins] {verdict.key_moment}")
            for s in verdict.scores:
                print(f"  Fighter {s.fighter}: logic={s.logic} evidence={s.evidence} "
                      f"rhetoric={s.rhetoric} responsiveness={s.responsiveness} "
                      f"total={s.total}  |  {s.justification}")

    # Closings
    print("\n--- CLOSINGS ---")
    closing_a, closing_b = run_closings(state)
    for closing in [closing_a, closing_b]:
        wc = len(closing.content.split())
        print(f"\nFIGHTER {closing.fighter} closing [{wc} words, {closing.seconds:.1f}s]:")
        print(closing.content)

    debate_elapsed = time.perf_counter() - debate_start
    a_total, b_total = cumulative_scores(state["verdicts"])

    print("\n" + "=" * 60)
    print(f"FINAL SCORES — A: {a_total}  B: {b_total}  Winner: {'A' if a_total > b_total else 'B' if b_total > a_total else 'TIE'}")
    print(f"\nTiming breakdown:")
    total_gen = 0.0
    for t in state["timings"]:
        print(f"  {t['kind']:15s} {t['model']:20s} {t['seconds']:.1f}s")
        total_gen += t["seconds"]
    print(f"\nTotal generation: {total_gen:.1f}s ({total_gen/60:.1f}min)")
    print(f"Total wall clock: {debate_elapsed:.1f}s ({debate_elapsed/60:.1f}min)")

    if total_gen > 420:  # 7 minutes
        print("\n⚠  OVER BUDGET: generation exceeded 7 minutes")
    else:
        print("\n✓ Within 7-minute generation budget")
