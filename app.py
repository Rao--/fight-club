"""
Streamlit UI + state machine driver. Only this file imports streamlit.
"""

import random
import time
from typing import Iterator

import streamlit as st

import config
import engine
import llm
import prompts
from schemas import BlockVerdict, Turn

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Model Fight Club",
    page_icon="🥊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def init_state() -> None:
    if "initialised" in st.session_state:
        return
    st.session_state.initialised = True
    st.session_state.phase = "SETUP"
    st.session_state.topic = ""
    st.session_state.exchange_num = 1
    st.session_state.turns = []
    st.session_state.verdicts = []
    st.session_state.fighter_a = None
    st.session_state.fighter_b = None
    st.session_state.judge_model = config.JUDGE_MODEL
    st.session_state.speaks_first = "A"
    st.session_state.timings = []
    st.session_state.error = None
    st.session_state.warm = False
    st.session_state.custom_persona_a = ""
    st.session_state.custom_persona_b = ""
    st.session_state.final_verdict_text = None


init_state()


def _format_seconds(s: float) -> str:
    m = int(s) // 60
    sec = int(s) % 60
    return f"{m}m {sec:02d}s" if m else f"{sec}s"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🥊 Model Fight Club")

    # --- Ollama health status ---
    ok, health_msg, available_tags = llm.health_check()
    if ok:
        st.success(health_msg, icon="✅")
    else:
        st.error(health_msg, icon="🔴")
        st.stop()

    # VRAM estimate warning
    if st.session_state.phase == "SETUP":
        st.divider()
        st.subheader("Setup")

        in_setup = True
    else:
        in_setup = False

    if in_setup:
        # ⭐ Special roast battle topic — prominent at top
        if st.button(
            "⭐ WHO IS THE BETTER AI? — roast battle",
            key="tp_roast",
            use_container_width=True,
            help="Each model brags about itself and roasts the opponent. Playground trash-talk energy.",
        ):
            st.session_state.topic = config.SELFROAST_TOPIC
            st.session_state["topic_input"] = config.SELFROAST_TOPIC

        # Scrollable preset list
        st.caption("or pick a topic:")
        with st.container(height=165, border=True):
            for i, topic in enumerate(config.TOPIC_PRESETS):
                short = topic.replace("This house believes that ", "").rstrip(".")
                short = short[0].upper() + short[1:]
                if st.button(short, key=f"tp_{i}", use_container_width=True, help=topic):
                    st.session_state.topic = topic
                    st.session_state["topic_input"] = topic

        # Initialise key on first render so widget picks it up without conflict
        if "topic_input" not in st.session_state:
            st.session_state["topic_input"] = st.session_state.topic
        topic_input = st.text_area("Motion (edit freely)", height=68, key="topic_input")
        st.session_state.topic = topic_input.strip()

        st.divider()

        # Fighter A
        st.subheader("Fighter A (FOR)")
        a_model_tags = [m.tag for m in config.FIGHTER_MODELS]
        a_model_labels = [m.label for m in config.FIGHTER_MODELS]
        a_default_idx = next(
            (i for i, m in enumerate(config.FIGHTER_MODELS) if m.tag == config.DEFAULT_FIGHTER_A_TAG), 0
        )
        a_model_sel = st.selectbox(
            "Model A",
            options=a_model_tags,
            format_func=lambda t: next(m.label for m in config.FIGHTER_MODELS if m.tag == t),
            index=a_default_idx,
            key="a_model_sel",
        )
        a_model_spec = config.MODELS_BY_TAG[a_model_sel]
        a_available = a_model_sel in available_tags
        if not a_available:
            st.warning(f"`ollama pull {a_model_sel}`", icon="⬇️")
        else:
            st.caption(a_model_spec.notes)

        a_persona_keys = [p.key for p in config.PERSONAS] + [config.CUSTOM_PERSONA_KEY]
        a_persona_labels = [p.name for p in config.PERSONAS] + ["Custom…"]
        a_persona_sel = st.selectbox("Persona A", options=a_persona_keys,
                                     format_func=lambda k: next(
                                         (p.name for p in config.PERSONAS if p.key == k), "Custom…"
                                     ),
                                     key="a_persona_sel")
        if a_persona_sel == config.CUSTOM_PERSONA_KEY:
            st.session_state.custom_persona_a = st.text_area(
                "Describe Fighter A's persona", value=st.session_state.custom_persona_a,
                height=80, key="custom_a_ta"
            )
            a_persona = config.Persona(
                key="custom",
                name="Custom Fighter",
                system_prompt=st.session_state.custom_persona_a,
                avatar="🤖",
            )
        else:
            a_persona = config.PERSONAS_BY_KEY[a_persona_sel]
        st.caption(f"{a_persona.avatar} {a_persona.name}")

        st.divider()

        # Fighter B
        st.subheader("Fighter B (AGAINST)")
        b_default_idx = next(
            (i for i, m in enumerate(config.FIGHTER_MODELS) if m.tag == config.DEFAULT_FIGHTER_B_TAG), 1
        )
        b_model_sel = st.selectbox(
            "Model B",
            options=a_model_tags,
            format_func=lambda t: next(m.label for m in config.FIGHTER_MODELS if m.tag == t),
            index=b_default_idx,
            key="b_model_sel",
        )
        b_model_spec = config.MODELS_BY_TAG[b_model_sel]
        b_available = b_model_sel in available_tags
        if not b_available:
            st.warning(f"`ollama pull {b_model_sel}`", icon="⬇️")
        else:
            st.caption(b_model_spec.notes)

        b_persona_sel = st.selectbox("Persona B", options=a_persona_keys,
                                     format_func=lambda k: next(
                                         (p.name for p in config.PERSONAS if p.key == k), "Custom…"
                                     ),
                                     index=1,
                                     key="b_persona_sel")
        if b_persona_sel == config.CUSTOM_PERSONA_KEY:
            st.session_state.custom_persona_b = st.text_area(
                "Describe Fighter B's persona", value=st.session_state.custom_persona_b,
                height=80, key="custom_b_ta"
            )
            b_persona = config.Persona(
                key="custom",
                name="Custom Fighter",
                system_prompt=st.session_state.custom_persona_b,
                avatar="🤖",
            )
        else:
            b_persona = config.PERSONAS_BY_KEY[b_persona_sel]
        st.caption(f"{b_persona.avatar} {b_persona.name}")

        st.divider()

        # Judge
        st.subheader("Judge")
        judge_available = config.JUDGE_MODEL in available_tags
        if not judge_available:
            # Demo fallback: prefer qwen2.5:3b, else any non-cloud non-fighter model
            fighter_tags = {a_model_sel, b_model_sel}
            local_tags = [t for t in available_tags if "cloud" not in t and t not in fighter_tags]
            # Prefer the smallest demo model
            preferred = [t for t in ["qwen2.5:3b"] if t in local_tags]
            fallback_judge = (preferred or local_tags or [None])[0]
            if fallback_judge:
                st.session_state.judge_model = fallback_judge
                st.warning(
                    f"Real judge `{config.JUDGE_MODEL}` not pulled yet. "
                    f"Using `{fallback_judge}` as demo judge.",
                    icon="⚖️",
                )
                judge_available = True
            else:
                st.warning(f"`ollama pull {config.JUDGE_MODEL}`", icon="⬇️")
        else:
            st.session_state.judge_model = config.JUDGE_MODEL
            st.caption(f"🏛️ {config.JUDGE_SPEC.label} — fixed, do not change")

        # Format info — static, no widgets
        st.divider()
        st.caption("**Format (fixed):** 3 rounds · 5 exchanges per round · 28 words per turn · judge scores each round")

        # VRAM estimate
        total_vram = a_model_spec.approx_vram_gb + b_model_spec.approx_vram_gb + config.JUDGE_SPEC.approx_vram_gb
        if total_vram > config.VRAM_WARNING_GB:
            st.warning(
                f"Estimated VRAM: ~{total_vram:.0f}GB — exceeds {config.VRAM_WARNING_GB:.0f}GB threshold. "
                "KV cache may be tight.",
                icon="⚠️",
            )
        else:
            st.info(f"Estimated VRAM: ~{total_vram:.0f}GB", icon="💾")

        # Start fight button
        models_differ = a_model_sel != b_model_sel
        demo_same_model = not models_differ and a_model_sel == "qwen2.5:3b"
        can_start = (
            bool(st.session_state.topic)
            and (models_differ or demo_same_model)
            and a_available
            and b_available
            and judge_available
        )

        if not models_differ and not demo_same_model:
            st.error("Fighter A and B must use different models.", icon="⚠️")
        elif demo_same_model:
            st.info("Demo mode: same model for both fighters. Pull real models for distinct styles.", icon="🎭")

        if st.button("🥊 Start fight", disabled=not can_start, type="primary", use_container_width=True):
            st.session_state.fighter_a = {"model": a_model_spec, "persona": a_persona}
            st.session_state.fighter_b = {"model": b_model_spec, "persona": b_persona}
            st.session_state.speaks_first = random.choice(["A", "B"])
            st.session_state.phase = "WARMING"
            st.rerun()

    else:
        # Post-setup — show summary and reset button
        if st.session_state.fighter_a:
            fa = st.session_state.fighter_a
            fb = st.session_state.fighter_b
            st.caption(f"**A:** {fa['model'].label} as {fa['persona'].name}")
            st.caption(f"**B:** {fb['model'].label} as {fb['persona'].name}")
            st.caption(f"**Judge:** {config.JUDGE_SPEC.label}")
        st.divider()
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Timing display (always visible after start)
    if st.session_state.timings:
        st.divider()
        total_gen = sum(t["seconds"] for t in st.session_state.timings)
        st.caption(f"Generation: {_format_seconds(total_gen)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_state() -> dict:
    """Package session_state into the dict engine.py expects."""
    return {
        "topic": st.session_state.topic,
        "fighter_a": st.session_state.fighter_a,
        "fighter_b": st.session_state.fighter_b,
        "judge_model": st.session_state.judge_model,
        "exchange_num": st.session_state.exchange_num,
        "turns": st.session_state.turns,
        "verdicts": st.session_state.verdicts,
        "timings": st.session_state.timings,
        "speaks_first": st.session_state.speaks_first,
    }


def _sync_from_engine_state(state: dict) -> None:
    st.session_state.turns = state["turns"]
    st.session_state.verdicts = state["verdicts"]
    st.session_state.timings = state["timings"]
    st.session_state.speaks_first = state["speaks_first"]


def _momentum_bar(verdicts: list[BlockVerdict]) -> None:
    """
    Renders a momentum bar using cumulative raw scores.
    Uses st.html with a flexbox div.
    """
    fa = st.session_state.fighter_a
    fb = st.session_state.fighter_b
    a_name = f"{fa['persona'].avatar} {fa['persona'].name}"
    b_name = f"{fb['persona'].avatar} {fb['persona'].name}"

    a_total, b_total = engine.cumulative_scores(verdicts)
    total = a_total + b_total

    if total == 0:
        a_pct = 50
        b_pct = 50
        greyed = True
    else:
        a_pct = round(100 * a_total / total)
        b_pct = 100 - a_pct
        greyed = False

    opacity = "0.4" if greyed else "1"
    a_color = f"rgba(91, 127, 166, {opacity})"    # muted slate blue
    b_color = f"rgba(123, 95, 166, {opacity})"    # muted terracotta

    html = f"""
    <div style="font-family: sans-serif; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-weight: bold; color: #5B7FA6;">{a_name} (FOR)</span>
            <span style="font-weight: bold; color: #7B5FA6;">(AGAINST) {b_name}</span>
        </div>
        <div style="display: flex; height: 32px; border-radius: 8px; overflow: hidden; border: 1px solid #444;">
            <div style="width: {a_pct}%; background: {a_color}; display: flex; align-items: center;
                        justify-content: center; color: white; font-weight: bold; font-size: 0.9rem;">
                {a_total if not greyed else "–"}
            </div>
            <div style="width: {b_pct}%; background: {b_color}; display: flex; align-items: center;
                        justify-content: center; color: white; font-weight: bold; font-size: 0.9rem;">
                {b_total if not greyed else "–"}
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 4px;
                    font-size: 0.8rem; color: #888;">
            <span>{a_pct}%</span>
            <span>{"50/50 — no blocks scored yet" if greyed else ""}</span>
            <span>{b_pct}%</span>
        </div>
    </div>
    """
    st.html(html)


def _verdict_card_html(v: BlockVerdict, fa: dict, fb: dict) -> str:
    """
    Returns a self-contained HTML card for one block verdict.
    Uses <details>/<summary> for collapsing — no JS needed.
    All CSS is inline so it works inside st.html() iframes.
    """
    A_COLOR = "#5B7FA6"
    B_COLOR = "#7B5FA6"
    winner_map = {
        "A": f"{fa['persona'].name} (A)",
        "B": f"{fb['persona'].name} (B)",
        "tie": "Tie",
    }
    winner_label = winner_map[v.block_winner]
    a_score = next(s for s in v.scores if s.fighter == "A")
    b_score = next(s for s in v.scores if s.fighter == "B")

    def score_rows(score, color: str) -> str:
        rows = ""
        for label, val in [
            ("Logic", score.logic),
            ("Evidence", score.evidence),
            ("Rhetoric", score.rhetoric),
            ("Responsive", score.responsiveness),
        ]:
            rows += (
                f"<div style='display:flex;align-items:center;gap:8px;margin:5px 0;"
                f"font-size:0.72rem;'>"
                f"<span style='width:72px;color:#888;flex-shrink:0;'>{label}</span>"
                f"<div style='flex:1;height:3px;background:rgba(128,128,128,0.15);border-radius:2px;'>"
                f"<div style='width:{val * 10}%;height:100%;background:{color};"
                f"border-radius:2px;transition:width 0.3s;'></div></div>"
                f"<span style='width:14px;text-align:right;font-weight:700;font-size:0.75rem;'>"
                f"{val}</span>"
                f"</div>"
            )
        return rows

    def fighter_col(score, cfg: dict, color: str) -> str:
        return (
            f"<div style='padding:14px 16px;'>"
            f"<div style='font-size:0.73rem;font-weight:700;color:{color};margin-bottom:10px;'>"
            f"{cfg['persona'].avatar} {cfg['persona'].name}</div>"
            f"{score_rows(score, color)}"
            f"<div style='margin-top:10px;padding-top:8px;"
            f"border-top:1px solid rgba(128,128,128,0.12);"
            f"display:flex;justify-content:space-between;font-size:0.78rem;'>"
            f"<span style='color:#888;'>Total</span>"
            f"<span style='font-weight:700;color:{color};font-size:0.9rem;'>{score.total}</span>"
            f"</div>"
            f"<div style='font-size:0.68rem;color:#888;font-style:italic;margin-top:8px;"
            f"line-height:1.45;'>{score.justification}</div>"
            f"</div>"
        )

    winner_bg = A_COLOR if v.block_winner == "A" else (B_COLOR if v.block_winner == "B" else "#666")

    return (
        f"<div style='border-radius:10px;overflow:hidden;"
        f"border:1px solid rgba(128,128,128,0.12);margin:14px 0;"
        f"font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;'>"
        f"<details>"
        f"<summary style='display:flex;align-items:center;gap:10px;padding:11px 16px;"
        f"cursor:pointer;background:rgba(128,128,128,0.04);list-style:none;"
        f"-webkit-appearance:none;'>"
        f"<span style='font-size:0.6rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:#888;flex-shrink:0;'>Round {v.block_num}</span>"
        f"<span style='background:{winner_bg}22;color:{winner_bg};border-radius:20px;"
        f"padding:2px 10px;font-size:0.68rem;font-weight:700;white-space:nowrap;'>"
        f"{winner_label}</span>"
        f"<span style='font-size:0.75rem;color:#999;flex:1;overflow:hidden;"
        f"text-overflow:ellipsis;white-space:nowrap;'>{v.key_moment}</span>"
        f"<span style='font-size:0.65rem;color:#666;flex-shrink:0;'>▸</span>"
        f"</summary>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;"
        f"border-top:1px solid rgba(128,128,128,0.08);'>"
        f"<div style='border-right:1px solid rgba(128,128,128,0.08);'>"
        f"{fighter_col(a_score, fa, A_COLOR)}</div>"
        f"{fighter_col(b_score, fb, B_COLOR)}"
        f"</div>"
        f"</details>"
        f"</div>"
    )


def _render_transcript() -> None:
    """Renders the full transcript with exchange dividers and block expanders."""
    turns = st.session_state.turns
    verdicts = st.session_state.verdicts
    fa = st.session_state.fighter_a
    fb = st.session_state.fighter_b

    # Group non-closing turns by exchange
    debate_turns = [t for t in turns if not t.is_closing]
    exchange_nums = sorted({t.exchange_num for t in debate_turns})

    # Build a lookup: exchange → verdict (the one that scores it)
    exchange_to_verdict: dict[int, BlockVerdict] = {}
    for v in verdicts:
        for ex in v.covers_exchanges:
            exchange_to_verdict[ex] = v

    last_exchange_in_block: dict[int, int] = {}  # block_num → last exchange it covers
    for v in verdicts:
        last_exchange_in_block[v.block_num] = max(v.covers_exchanges)

    shown_rounds: set = set()
    for ex_num in exchange_nums:
        rn = (ex_num - 1) // config.EXCHANGES_PER_ROUND + 1
        if rn not in shown_rounds:
            shown_rounds.add(rn)
            st.markdown(
                f"<div style='text-align:center; font-weight:bold; font-size:1rem;"
                f" color:#888; margin:1rem 0 0.3rem;'>— Round {rn} —</div>",
                unsafe_allow_html=True,
            )

        round_ex = (ex_num - 1) % config.EXCHANGES_PER_ROUND + 1
        st.markdown(
            f"<div style='text-align:center; color:#aaa; font-size:0.72rem; margin:0.1rem 0;'>"
            f"· exchange {round_ex} of {config.EXCHANGES_PER_ROUND} ·</div>",
            unsafe_allow_html=True,
        )
        ex_turns = [t for t in debate_turns if t.exchange_num == ex_num]
        for turn in ex_turns:
            cfg = fa if turn.fighter == "A" else fb
            is_b = turn.fighter == "B"
            wc = len(turn.content.split())
            bubble_bg = "#5B7FA6" if not is_b else "#7B5FA6"
            align = "flex-end" if is_b else "flex-start"
            text_align = "right" if is_b else "left"
            label = f"{cfg['persona'].avatar} {cfg['persona'].name}"
            meta_parts = [f"{wc}w"]
            if turn.tokens_out:
                meta_parts.append(f"{turn.tokens_out} tok out")
            if turn.tokens_in:
                meta_parts.append(f"{turn.tokens_in} in")
            if turn.tps:
                meta_parts.append(f"{turn.tps:.1f} tok/s")
            if turn.ttft:
                meta_parts.append(f"TTFT {turn.ttft:.2f}s")
            meta_parts.append(f"{turn.seconds:.1f}s")
            meta = " · ".join(meta_parts)
            name_side = f"<div style='text-align:{text_align};font-size:0.75rem;color:#888;margin-bottom:2px;'>{label}</div>"
            bubble = (
                f"<div style='display:flex;justify-content:{align};margin:4px 0;'>"
                f"<div style='max-width:75%;background:{bubble_bg};color:#fff;"
                f"border-radius:12px;padding:10px 14px;font-size:0.9rem;line-height:1.5;'>"
                f"{turn.content}"
                f"<div style='font-size:0.6rem;opacity:0.55;margin-top:6px;"
                f"text-align:{text_align};font-family:monospace;letter-spacing:0.01em;'>{meta}</div>"
                f"</div></div>"
            )
            st.html(name_side + bubble)

        # After last exchange of a round, show verdict card
        v = exchange_to_verdict.get(ex_num)
        if v and max(v.covers_exchanges) == ex_num:
            st.html(_verdict_card_html(v, fa, fb))

    # Closing statements
    closings = [t for t in turns if t.is_closing]
    if closings:
        st.divider()
        st.markdown("#### Closing Statements *(unscored)*")
        cols = st.columns(2)
        for closing in sorted(closings, key=lambda t: t.fighter):
            cfg = fa if closing.fighter == "A" else fb
            with cols[0 if closing.fighter == "A" else 1]:
                st.markdown(f"**{cfg['persona'].avatar} {cfg['persona'].name}** (Fighter {closing.fighter})")
                st.write(closing.content)
                wc = len(closing.content.split())
                st.caption(f"{wc} words · {closing.seconds:.1f}s · Unscored")


def _export_markdown() -> str:
    """Generates full transcript + verdicts + timings as markdown."""
    lines = [
        "# Model Fight Club — Debate Transcript",
        "",
        f"**Motion:** {st.session_state.topic}",
        "",
    ]
    fa = st.session_state.fighter_a
    fb = st.session_state.fighter_b
    lines += [
        f"**Fighter A:** {fa['model'].label} as {fa['persona'].name}",
        f"**Fighter B:** {fb['model'].label} as {fb['persona'].name}",
        f"**Judge:** {st.session_state.judge_model}",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]

    debate_turns = [t for t in st.session_state.turns if not t.is_closing]
    for ex_num in sorted({t.exchange_num for t in debate_turns}):
        lines.append(f"### Exchange {ex_num}")
        for turn in [t for t in debate_turns if t.exchange_num == ex_num]:
            cfg = fa if turn.fighter == "A" else fb
            lines += [
                f"**Fighter {turn.fighter} — {cfg['persona'].name}** ({cfg['model'].tag}, {turn.seconds:.1f}s)",
                "",
                turn.content,
                "",
            ]

    closings = [t for t in st.session_state.turns if t.is_closing]
    if closings:
        lines += ["---", "", "## Closing Statements *(unscored)*", ""]
        for c in sorted(closings, key=lambda t: t.fighter):
            cfg = fa if c.fighter == "A" else fb
            lines += [
                f"**Fighter {c.fighter} — {cfg['persona'].name}**",
                "",
                c.content,
                "",
            ]

    lines += ["---", "", "## Verdicts", ""]
    for v in st.session_state.verdicts:
        lines.append(f"### Block {v.block_num} (exchanges {v.covers_exchanges}) — **{v.block_winner.upper()}** wins")
        lines.append(f"*{v.key_moment}*")
        lines.append("")
        lines.append("| Fighter | Logic | Evidence | Rhetoric | Responsiveness | Total | Note |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in v.scores:
            lines.append(f"| {s.fighter} | {s.logic} | {s.evidence} | {s.rhetoric} | {s.responsiveness} | {s.total} | {s.justification} |")
        lines.append("")

    a_total, b_total = engine.cumulative_scores(st.session_state.verdicts)
    winner = "A" if a_total > b_total else "B" if b_total > a_total else "Tie"
    lines += [
        "---",
        "",
        "## Final Scores",
        "",
        f"- Fighter A: **{a_total}**",
        f"- Fighter B: **{b_total}**",
        f"- **Winner: {winner}**",
        "",
        "---",
        "",
        "## Timings",
        "",
        "| Kind | Model | Seconds |",
        "|---|---|---|",
    ]
    total_gen = 0.0
    for t in st.session_state.timings:
        lines.append(f"| {t['kind']} | {t['model']} | {t['seconds']:.1f} |")
        total_gen += t["seconds"]
    lines += [
        "",
        f"**Total generation time:** {total_gen:.1f}s ({total_gen/60:.1f}min)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------------

# Error banner
if st.session_state.error:
    st.error(st.session_state.error, icon="🚨")

# ---------------------------------------------------------------------------
# WARMING phase
# ---------------------------------------------------------------------------

if st.session_state.phase == "WARMING":
    st.header("Warming up models…")
    st.caption("Loading three models into VRAM. First cold load of the 27B judge takes 20–30s.")

    fa = st.session_state.fighter_a
    fb = st.session_state.fighter_b
    warm_order = [
        config.JUDGE_MODEL,
        fa["model"].tag,
        fb["model"].tag,
    ]

    progress = st.progress(0, text="Starting warmup…")
    status_lines = []

    t0 = time.perf_counter()
    for i, tag in enumerate(warm_order):
        progress.progress((i) / len(warm_order), text=f"Loading {tag}…")
        try:
            t_start = time.perf_counter()
            llm.warmup([tag])
            elapsed = time.perf_counter() - t_start
            status_lines.append(f"✅ {tag} — {elapsed:.1f}s")
            if elapsed > config.WARMUP_CEILING_S:
                st.warning(f"{tag} took {elapsed:.1f}s (ceiling {config.WARMUP_CEILING_S}s)", icon="⏱")
        except Exception as e:
            st.error(f"Warmup failed for {tag}: {e}")
            st.session_state.error = f"Warmup failed for {tag}: {e}"

        progress.progress((i + 1) / len(warm_order), text=f"Loaded {tag}")
        for line in status_lines:
            st.caption(line)

    # Verify all three resident
    _, _, resident_tags = llm.health_check()
    missing_resident = [t for t in warm_order if not any(t in r for r in resident_tags)]
    if missing_resident:
        st.warning(
            f"Models not confirmed resident: {missing_resident}. "
            "Check `OLLAMA_MAX_LOADED_MODELS=3`. Expect slow turns if models are swapping.",
            icon="⚠️",
        )

    total_warmup = time.perf_counter() - t0
    st.session_state.timings.append({"kind": "warmup", "model": "all", "seconds": total_warmup})
    st.session_state.warm = True
    st.session_state.phase = "EXCHANGE"
    st.rerun()

# ---------------------------------------------------------------------------
# EXCHANGE phase
# ---------------------------------------------------------------------------

elif st.session_state.phase == "EXCHANGE":
    ex_num = st.session_state.exchange_num
    round_num = (ex_num - 1) // config.EXCHANGES_PER_ROUND + 1

    is_roast = st.session_state.topic == config.SELFROAST_TOPIC
    fa_tag = st.session_state.fighter_a["model"].tag if st.session_state.fighter_a else ""
    fb_tag = st.session_state.fighter_b["model"].tag if st.session_state.fighter_b else ""
    st.header("Model Fight Club", divider="red")
    if is_roast:
        st.caption(f"⭐ **AI Roast Battle** — {fa_tag} vs {fb_tag} — who is better?")
    else:
        st.caption(f"**Motion:** {st.session_state.topic}")

    _momentum_bar(st.session_state.verdicts)
    _render_transcript()

    total_gen = sum(t["seconds"] for t in st.session_state.timings)
    col_btn, col_time = st.columns([3, 1])

    with col_btn:
        label = f"▶ Round {round_num}" if round_num == 1 else f"Round {round_num} →"
        if st.button(label, type="primary", use_container_width=True):
            fa = st.session_state.fighter_a
            fb = st.session_state.fighter_b
            state = _engine_state()

            def _build_messages_for_stream(st8: dict, fighter_letter: str) -> list[dict]:
                cfg = fa if fighter_letter == "A" else fb
                opp_cfg = fb if fighter_letter == "A" else fa

                # ⭐ Roast battle — each model argues it is the better AI
                if st8["topic"] == config.SELFROAST_TOPIC:
                    system = prompts.fighter_roast_system(
                        my_model_tag=cfg["model"].tag,
                        opponent_model_tag=opp_cfg["model"].tag,
                    )
                    user = prompts.fighter_roast_user(
                        exchange_num=st8["exchange_num"],
                        turns=[t for t in st8["turns"] if not t.is_closing],
                        my_model_tag=cfg["model"].tag,
                        opponent_model_tag=opp_cfg["model"].tag,
                        fighter_letter=fighter_letter,
                    )
                    return [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]

                # Normal debate
                persona = cfg["persona"]
                position = "FOR" if fighter_letter == "A" else "AGAINST"
                a_total, b_total = engine.cumulative_scores(st8["verdicts"])
                aggressive = False
                ex = st8["exchange_num"]
                round_ex = (ex - 1) % config.EXCHANGES_PER_ROUND + 1
                if round_ex >= 3:
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
                    exchange_num=st8["exchange_num"],
                    turns=[t for t in st8["turns"] if not t.is_closing],
                    persona_name=persona.name,
                    fighter_letter=fighter_letter,
                    aggressive=aggressive,
                )
                return [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]

            # Show round header
            st.markdown(
                f"<div style='text-align:center; font-weight:bold; font-size:1.1rem;"
                f" color:#888; margin:1rem 0 0.5rem;'>— Round {round_num} —</div>",
                unsafe_allow_html=True,
            )

            # Run all 5 exchanges in this round back-to-back
            round_start_ex = ex_num
            round_end_ex = round_start_ex + config.EXCHANGES_PER_ROUND - 1

            for ex in range(round_start_ex, round_end_ex + 1):
                state["exchange_num"] = ex
                speaks_first = state["speaks_first"]
                speaks_second = "B" if speaks_first == "A" else "A"

                st.markdown(
                    f"<div style='text-align:center; color:#888; font-size:0.75rem;"
                    f" margin:0.25rem 0;'>· exchange {(ex - 1) % config.EXCHANGES_PER_ROUND + 1} of {config.EXCHANGES_PER_ROUND} ·</div>",
                    unsafe_allow_html=True,
                )

                for fighter_letter in [speaks_first, speaks_second]:
                    cfg = fa if fighter_letter == "A" else fb
                    model_tag = cfg["model"].tag
                    persona = cfg["persona"]
                    messages = _build_messages_for_stream(state, fighter_letter)
                    is_right = fighter_letter == "B"
                    align = "flex-end" if is_right else "flex-start"
                    text_align = "right" if is_right else "left"
                    bubble_bg = "#7B5FA6" if is_right else "#5B7FA6"
                    label = f"{persona.avatar} {persona.name}"

                    st.markdown(
                        f"<div style='text-align:{text_align};font-size:0.75rem;"
                        f"color:#888;margin-bottom:2px;'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                    placeholder = st.empty()
                    gen = llm.chat_stream(
                        model=model_tag,
                        messages=messages,
                        num_predict=config.FIGHTER_NUM_PREDICT,
                        temperature=config.FIGHTER_TEMPERATURE,
                        think=False,
                    )
                    content = ""
                    chunk_count = 0
                    first_token_t = None
                    t_stream_start = time.perf_counter()
                    for chunk in gen:
                        content += chunk
                        chunk_count += 1
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                        dt = time.perf_counter() - first_token_t
                        live_tps = chunk_count / dt if dt > 0 else 0.0
                        placeholder.markdown(
                            f"<div style='display:flex;justify-content:{align};margin:4px 0;'>"
                            f"<div style='max-width:75%;background:{bubble_bg};color:#fff;"
                            f"border-radius:12px;padding:10px 14px;font-size:0.9rem;line-height:1.5;'>"
                            f"{content}▌"
                            f"<div style='font-size:0.6rem;opacity:0.45;margin-top:3px;"
                            f"text-align:{text_align};font-family:monospace;'>"
                            f"{chunk_count} tok · {live_tps:.0f}/s</div>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )
                    elapsed = getattr(llm.chat_stream, "_last_duration", time.perf_counter() - t_stream_start)
                    ttft = getattr(llm.chat_stream, "_last_ttft", 0.0)
                    tokens_out = getattr(llm.chat_stream, "_last_tokens_out", chunk_count)
                    tokens_in = getattr(llm.chat_stream, "_last_tokens_in", 0)
                    tps = getattr(llm.chat_stream, "_last_tps", 0.0)
                    wc = len(content.split())
                    metrics = (
                        f"{wc}w · {tokens_out} tok out"
                        + (f" / {tokens_in} in" if tokens_in else "")
                        + (f" · {tps:.1f} tok/s" if tps else "")
                        + (f" · TTFT {ttft:.2f}s" if ttft else "")
                        + f" · {elapsed:.1f}s"
                    )
                    placeholder.markdown(
                        f"<div style='display:flex;justify-content:{align};margin:4px 0;'>"
                        f"<div style='max-width:75%;background:{bubble_bg};color:#fff;"
                        f"border-radius:12px;padding:10px 14px;font-size:0.9rem;line-height:1.5;'>"
                        f"{content}"
                        f"<div style='font-size:0.6rem;opacity:0.55;margin-top:6px;"
                        f"text-align:{text_align};font-family:monospace;letter-spacing:0.01em;'>"
                        f"{metrics}</div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                    if elapsed > config.FIGHTER_CEILING_S:
                        st.warning(f"Fighter {fighter_letter} ({model_tag}): {elapsed:.1f}s over ceiling", icon="⏱")

                    state["turns"].append(Turn(
                        exchange_num=ex,
                        fighter=fighter_letter,
                        model=model_tag,
                        persona_name=persona.name,
                        content=content.strip(),
                        seconds=elapsed,
                        tokens_out=tokens_out,
                        tokens_in=tokens_in,
                        tps=tps,
                        ttft=ttft,
                    ))
                    state["timings"].append({"kind": "fighter", "model": model_tag, "seconds": elapsed})

                # Do NOT randomize within the round — keep speaks_first constant so
                # turns always strictly alternate A,B,A,B with no back-to-back same person

            # Flip who opens next round (variety across rounds, still no consecutive same person)
            state["speaks_first"] = "B" if state["speaks_first"] == "A" else "A"

            # Judge this round
            state["exchange_num"] = round_end_ex
            with st.spinner(f"Judge scoring Round {round_num}…"):
                verdict = engine._run_judge(state)
            if verdict:
                state["verdicts"].append(verdict)
                winner_label = {
                    "A": f"{fa['persona'].name} (A)",
                    "B": f"{fb['persona'].name} (B)",
                    "tie": "Tie",
                }[verdict.block_winner]
                st.success(f"Round {round_num}: **{winner_label}** — {verdict.key_moment}", icon="⚖️")

            _sync_from_engine_state(state)
            st.session_state.exchange_num = round_end_ex + 1

            if st.session_state.exchange_num > config.MAX_EXCHANGES:
                st.session_state.phase = "CLOSING"

            st.rerun()

    with col_time:
        st.caption(f"Generation: {_format_seconds(total_gen)}")

# ---------------------------------------------------------------------------
# CLOSING phase
# ---------------------------------------------------------------------------

elif st.session_state.phase == "CLOSING":
    st.header("Model Fight Club", divider="red")
    st.caption(f"**Motion:** {st.session_state.topic}")

    _momentum_bar(st.session_state.verdicts)
    _render_transcript()

    total_gen = sum(t["seconds"] for t in st.session_state.timings)
    col_btn, col_time = st.columns([3, 1])

    with col_btn:
        if st.button("Closing statements", type="primary", use_container_width=True):
            fa = st.session_state.fighter_a
            fb = st.session_state.fighter_b

            debate_turns = [t for t in st.session_state.turns if not t.is_closing]

            with st.spinner("Generating closing statements (both fighters, blind)…"):
                for fighter_letter in ["A", "B"]:
                    cfg = fa if fighter_letter == "A" else fb
                    model_tag = cfg["model"].tag
                    persona = cfg["persona"]
                    position = "FOR" if fighter_letter == "A" else "AGAINST"

                    system = prompts.fighter_system(
                        persona_name=persona.name,
                        persona_system_prompt=persona.system_prompt,
                        fighter_letter=fighter_letter,
                        topic=st.session_state.topic,
                        position=position,
                    )
                    user = prompts.closing_user(
                        topic=st.session_state.topic,
                        turns=debate_turns,
                        persona_name=persona.name,
                    )
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]

                    content, elapsed = llm.chat(
                        model=model_tag,
                        messages=messages,
                        num_predict=config.CLOSING_NUM_PREDICT,
                        temperature=config.FIGHTER_TEMPERATURE,
                    )

                    if elapsed > config.CLOSING_CEILING_S:
                        st.warning(f"Closing {fighter_letter} ({model_tag}) took {elapsed:.1f}s", icon="⏱")

                    closing = Turn(
                        exchange_num=0,
                        fighter=fighter_letter,
                        model=model_tag,
                        persona_name=persona.name,
                        content=content.strip(),
                        seconds=elapsed,
                        is_closing=True,
                    )
                    st.session_state.turns.append(closing)
                    st.session_state.timings.append({"kind": "closing", "model": model_tag, "seconds": elapsed})

            # Final verdict from the judge
            a_total, b_total = engine.cumulative_scores(st.session_state.verdicts)
            with st.spinner("Judge writing final verdict…"):
                fv_messages = [
                    {"role": "system", "content": prompts.final_verdict_system()},
                    {"role": "user", "content": prompts.final_verdict_user(
                        topic=st.session_state.topic,
                        turns=st.session_state.turns,
                        verdicts=st.session_state.verdicts,
                        fighter_a_name=fa["persona"].name,
                        fighter_b_name=fb["persona"].name,
                        a_total=a_total,
                        b_total=b_total,
                    )},
                ]
                fv_text, fv_elapsed = llm.chat(
                    model=st.session_state.judge_model,
                    messages=fv_messages,
                    num_predict=config.FINAL_VERDICT_NUM_PREDICT,
                    think=config.JUDGE_THINK,
                    temperature=config.JUDGE_TEMPERATURE,
                )
                st.session_state.final_verdict_text = fv_text.strip()
                st.session_state.timings.append({
                    "kind": "final_verdict",
                    "model": st.session_state.judge_model,
                    "seconds": fv_elapsed,
                })

            st.session_state.phase = "FINISHED"
            st.rerun()

    with col_time:
        st.caption(f"Generation: {_format_seconds(total_gen)}")

# ---------------------------------------------------------------------------
# FINISHED phase
# ---------------------------------------------------------------------------

elif st.session_state.phase == "FINISHED":
    st.header("Model Fight Club — Final Verdict", divider="red")
    st.caption(f"**Motion:** {st.session_state.topic}")

    _momentum_bar(st.session_state.verdicts)

    a_total, b_total = engine.cumulative_scores(st.session_state.verdicts)
    fa = st.session_state.fighter_a
    fb = st.session_state.fighter_b

    if a_total > b_total:
        winner_str = f"🏆 Fighter A — {fa['persona'].name} ({fa['model'].label}) wins!"
        st.success(winner_str)
    elif b_total > a_total:
        winner_str = f"🏆 Fighter B — {fb['persona'].name} ({fb['model'].label}) wins!"
        st.success(winner_str)
    else:
        st.info("🤝 It's a tie!")

    # Judge's final verdict — styled narrative card
    if st.session_state.final_verdict_text:
        winner_color = "#5B7FA6" if a_total >= b_total else "#7B5FA6"
        st.html(
            f"<div style='border-radius:12px;border:1px solid {winner_color}44;"
            f"background:{winner_color}0d;padding:20px 22px;margin:10px 0 22px;"
            f"font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;'>"
            f"<div style='font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;"
            f"color:#888;margin-bottom:12px;font-weight:700;'>⚖️ Judge's Final Verdict</div>"
            f"<p style='font-size:0.96rem;line-height:1.72;margin:0 0 16px;'>"
            f"{st.session_state.final_verdict_text}</p>"
            f"<div style='display:flex;gap:20px;font-size:0.7rem;"
            f"border-top:1px solid rgba(128,128,128,0.12);padding-top:10px;'>"
            f"<span style='color:#5B7FA6;font-weight:600;'>"
            f"{fa['persona'].avatar} {fa['persona'].name}: {a_total} pts</span>"
            f"<span style='color:#7B5FA6;font-weight:600;'>"
            f"{fb['persona'].avatar} {fb['persona'].name}: {b_total} pts</span>"
            f"<span style='margin-left:auto;color:#666;font-size:0.65rem;'>"
            f"Judged by {st.session_state.judge_model}</span>"
            f"</div>"
            f"</div>"
        )

    # Round-by-round breakdown
    st.subheader("Round breakdown")
    for v in st.session_state.verdicts:
        st.html(_verdict_card_html(v, fa, fb))

    # Full transcript
    st.subheader("Full transcript")
    _render_transcript()

    # Generation timings
    st.subheader("Timings")
    total_gen = sum(t["seconds"] for t in st.session_state.timings)
    model_totals: dict[str, float] = {}
    for t in st.session_state.timings:
        model_totals[t["model"]] = model_totals.get(t["model"], 0.0) + t["seconds"]
    for model, secs in sorted(model_totals.items(), key=lambda x: -x[1]):
        st.caption(f"{model}: {secs:.1f}s")
    st.caption(f"**Total generation: {_format_seconds(total_gen)}**")

    # Export
    st.download_button(
        label="⬇️ Export transcript (markdown)",
        data=_export_markdown(),
        file_name="fight-club-transcript.md",
        mime="text/markdown",
        use_container_width=True,
    )
