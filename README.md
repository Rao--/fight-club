# Model Fight Club

Two local LLMs argue a topic across 15 short exchanges (3 rounds × 5). A third LLM judges each round on four criteria and produces a running momentum bar, styled verdict cards, and a final chairperson verdict. Everything runs on your machine via Ollama — no API keys, no cloud costs.

---

## What it does

- **3 rounds × 5 exchanges** — Fighter A argues FOR the motion, Fighter B AGAINST. Each turn is capped at 28 words.
- **6 built-in personas** — Cynical VC, Idealistic Grad Student, 1950s Ad Man, Modern UX Researcher, Retired Trial Lawyer, Overconfident Sports Commentator — plus a free-text custom persona.
- **5 selectable fighter models** — gemma4:12b, qwen3.5:9b, llama3.1:8b, mistral:7b, qwen2.5:3b (demo). Any two may be paired.
- **20 trending topic presets** across AI/tech, work/money, health/society, and environment/future — scrollable picker in the sidebar.
- **⭐ Roast battle mode** — a special topic where each model argues it is the better AI and roasts the opponent. Completely separate prompt path; anti-repetition enforcement injects the model's own previous turns verbatim so it cannot recycle arguments.
- **Real-time streaming bubbles** — Fighter A on the left (slate blue), Fighter B on the right (muted purple). A live `tok · tok/s` ticker updates on every token chunk during generation.
- **Production metrics on every bubble** — output tokens, input tokens, tokens/second (Ollama internal timer), TTFT, and wall time. Stored on each turn and re-rendered in the full transcript.
- **Judge scoring** — 4 criteria (logic, evidence, rhetoric, responsiveness), each 1-10, after every round. Collapsible verdict cards with CSS score bars. Parse-and-retry with Pydantic; fallback to neutral 5s if the judge fails twice.
- **Momentum bar** — proportional score bar updating after each judged block.
- **Final verdict** — judge recast as chairperson produces a 3-5 sentence narrative: declares winner, names decisive argument, acknowledges losing side's best moment.
- **Closing statements** — both fighters speak blind after exchange 15. Not scored.
- **Markdown export** — full transcript, block verdicts, scores, and timing table downloadable as `.md`.

---

## Requirements

- Python >= 3.11
- [Ollama](https://ollama.com) installed and running locally
- ~30 GB VRAM to keep all three models resident simultaneously (judge 17 GB + two fighters ~6-7 GB each). A demo run with `qwen2.5:3b` as both fighters needs far less.

---

## Required environment variables

Set these **before** starting Ollama:

```bash
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PARALLEL=1
```

---

## Model pulls

```bash
ollama pull qwen3.6:27b    # Judge — ~17 GB VRAM
ollama pull gemma4:12b     # Default Fighter A — ~7 GB
ollama pull qwen3.5:9b     # Default Fighter B — ~6 GB
ollama pull llama3.1:8b    # Optional — ~5 GB
ollama pull mistral:7b     # Optional — ~4 GB
ollama pull qwen2.5:3b     # Demo / low-VRAM fallback — ~2 GB
```

---

## Installation

```bash
cd fight-club
pip install -r requirements.txt
```

---

## Running

```bash
# Terminal 1 — Ollama with required env vars
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PARALLEL=1
ollama serve

# Terminal 2 — Streamlit app
streamlit run app.py
```

---

## Smoke tests

Run these before the UI to verify Ollama connectivity and model behaviour:

```bash
# CP1 — Ollama health check + streaming test
python llm.py

# CP2 + CP3 — headless debate + judge (all three models must be pulled)
python engine.py
```

`engine.py` prints a timing table at the end. Total generation should be under 7 minutes.

---

## Project layout

```
fight-club/
├── app.py          # Streamlit UI and state machine (SETUP→WARMING→EXCHANGE→CLOSING→FINISHED)
├── engine.py       # Debate orchestration — pure Python, no Streamlit
├── llm.py          # Ollama wrapper (health_check, chat, chat_stream, warmup)
├── prompts.py      # All prompt text — fighter, judge, roast battle, final verdict, closings
├── schemas.py      # Data model — Turn dataclass, BlockVerdict / TurnScore Pydantic models
├── config.py       # Model registry, persona registry, all numeric tunables
└── requirements.txt
```

Only `app.py` imports Streamlit. All other files are plain Python and can be tested in isolation.

---

## Debate format

| Setting | Value |
|---|---|
| Rounds | 3 |
| Exchanges per round | 5 |
| Total exchanges | 15 |
| Words per turn (target) | 28 |
| Judge fires after | Exchanges 5, 10, 15 |
| Closing statements | Yes — blind, unjudged |
| Clicks to complete a debate | 4 (3 round buttons + 1 closing button) |

---

## If it runs slow

Diagnose in this order:

1. `ollama ps` — all three models must show as resident. If only one or two appear, check `OLLAMA_MAX_LOADED_MODELS=3`.
2. `JUDGE_THINK = False` in `config.py` — gemma4 and qwen3.x default to thinking mode ON, routing all output to the thinking field and leaving content empty. This must be `False`.
3. `OLLAMA_NUM_PARALLEL = 1` — parallel requests cause model swapping and kill throughput.
