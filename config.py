"""
Model registry, persona registry, and tunables.
No Streamlit imports.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    label: str
    approx_vram_gb: float
    notes: str


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    system_prompt: str
    avatar: str


# ---------------------------------------------------------------------------
# Model registry — verify tags against `ollama list` before use
# ---------------------------------------------------------------------------

JUDGE_MODEL = "qwen3.6:27b"  # switch back once pulled

JUDGE_SPEC = ModelSpec(
    tag="qwen3.6:27b",
    label="Qwen 3.6 27B",
    approx_vram_gb=17.0,
    notes="Dense 27B, 256K context. Instruction-following judge; thinking mode OFF.",
)

FIGHTER_MODELS: list[ModelSpec] = [
    ModelSpec(
        tag="gemma4:12b",
        label="Gemma 4 12B",
        approx_vram_gb=7.0,
        notes="Verbose and warm; wants to explain itself. The 75-word cap fights it hardest.",
    ),
    ModelSpec(
        tag="qwen3.5:9b",
        label="Qwen 3.5 9B",
        approx_vram_gb=6.0,
        notes="Structured reasoner; reaches for enumeration. Rule 7 in the prompt exists for this.",
    ),
    ModelSpec(
        tag="llama3.1:8b",
        label="Llama 3.1 8B",
        approx_vram_gb=5.0,
        notes="Diplomatic, hedges, concedes too readily. High comedy value; biggest stress test of anti-capitulation rules.",
    ),
    ModelSpec(
        tag="mistral:7b",
        label="Mistral 7B",
        approx_vram_gb=4.0,
        notes="Blunt and terse. Naturally suited to the short-turn format.",
    ),
    # --- Demo model: available now, replace with real models after pulling ---
    ModelSpec(
        tag="qwen2.5:3b",
        label="Qwen 2.5 3B (demo)",
        approx_vram_gb=2.0,
        notes="Tiny stand-in for demo. Pull gemma4:12b / qwen3.5:9b for real debates.",
    ),
]

# Default pairing — warm rambler vs clipped enumerator
DEFAULT_FIGHTER_A_TAG = "gemma4:12b"
DEFAULT_FIGHTER_B_TAG = "qwen3.5:9b"

MODELS_BY_TAG: dict[str, ModelSpec] = {m.tag: m for m in FIGHTER_MODELS}

# ---------------------------------------------------------------------------
# Persona registry
# ---------------------------------------------------------------------------

PERSONAS: list[Persona] = [
    Persona(
        key="cynical_vc",
        name="Cynical VC",
        system_prompt=(
            "You speak like a sand-hill road venture capitalist who has seen it all. "
            "Your vocabulary is littered with 'runway', 'burn rate', 'moat', and 'TAM'. "
            "You are constitutionally suspicious of anything that cannot be monetised in 18 months."
        ),
        avatar="💰",
    ),
    Persona(
        key="idealistic_grad",
        name="Idealistic Grad Student",
        system_prompt=(
            "You speak like a second-year PhD student in social science who is still certain "
            "that rigorous research can fix anything. You cite studies (even approximate ones), "
            "pepper your speech with 'the literature suggests' and 'we need to problematise this', "
            "and treat anecdote as the enemy of knowledge."
        ),
        avatar="📚",
    ),
    Persona(
        key="ad_man_1950s",
        name="1950s Ad Man",
        system_prompt=(
            "You speak like a Madison Avenue copywriter from 1955 — confident, hyperbolic, and "
            "deeply certain that the right slogan can sell anything. Your reference points are "
            "television, the American dream, and 'what the consumer wants'. You never use the "
            "word 'algorithm'."
        ),
        avatar="🎩",
    ),
    Persona(
        key="ux_researcher",
        name="Modern UX Researcher",
        system_prompt=(
            "You speak like a senior UX researcher steeped in design thinking. "
            "You frame everything through user pain points, journey maps, and empathy. "
            "You use phrases like 'how might we', 'the mental model here', and 'the signal "
            "from our user interviews'. You treat data as provisional and context as everything."
        ),
        avatar="🔬",
    ),
    Persona(
        key="trial_lawyer",
        name="Retired Trial Lawyer",
        system_prompt=(
            "You speak like a retired trial attorney who has cross-examined hundreds of witnesses. "
            "Your sentences are tight and purposeful. You establish facts before making inferences, "
            "you catch logical gaps by name ('that's a non-sequitur', 'that assumes facts not in "
            "evidence'), and you know exactly when to pause for effect."
        ),
        avatar="⚖️",
    ),
    Persona(
        key="sports_commentator",
        name="Overconfident Sports Commentator",
        system_prompt=(
            "You speak like a sports commentator who has accidentally wandered into a debate. "
            "You use sports metaphors compulsively: 'playing offence', 'the pivot', 'hat trick', "
            "'fourth-quarter move'. You are always on the verge of declaring a winner regardless "
            "of the evidence, and you treat uncertainty as a sign of weakness."
        ),
        avatar="🏆",
    ),
]

PERSONAS_BY_KEY: dict[str, Persona] = {p.key: p for p in PERSONAS}
CUSTOM_PERSONA_KEY = "custom"

# ---------------------------------------------------------------------------
# Tunables — do not expose to the user as UI controls
# ---------------------------------------------------------------------------

NUM_ROUNDS = 3
EXCHANGES_PER_ROUND = 5
MAX_EXCHANGES = NUM_ROUNDS * EXCHANGES_PER_ROUND   # = 15 total
JUDGE_AFTER = {5, 10, 15}     # judge once after each round (every 5 exchanges)
TARGET_WORDS = 28
CLOSING_WORDS = 60

FIGHTER_NUM_PREDICT = 55       # headroom over 28-word target for a clean sentence ending
JUDGE_NUM_PREDICT = 400
CLOSING_NUM_PREDICT = 120
FINAL_VERDICT_NUM_PREDICT = 280

FIGHTER_TEMPERATURE = 0.9
JUDGE_TEMPERATURE = 0.2
JUDGE_THINK = False

FIGHTER_CEILING_S = 15.0
JUDGE_CEILING_S = 45.0
CLOSING_CEILING_S = 20.0
WARMUP_CEILING_S = 90.0

# VRAM warning threshold (GB) — sum of three model weights
VRAM_WARNING_GB = 36.0

# ---------------------------------------------------------------------------
# Topic presets
# ---------------------------------------------------------------------------

# Special self-roast topic — unique prompt, each model argues it is superior
SELFROAST_TOPIC = "⭐ WHO IS THE BETTER AI? Each model brags about itself and roasts the opponent — no gloves."

TOPIC_PRESETS: list[str] = [
    # 🔥 AI & Technology — trending now
    "This house believes that AI-generated content should carry a permanent, mandatory label.",
    "This house believes that AI in hiring is more biased than human recruiters.",
    "This house believes that social media platforms should be banned for users under 16.",
    "This house believes that AI will destroy more jobs than it creates in the next decade.",
    "This house believes that big tech companies know too much about us to be trusted with AI.",
    # 🔥 Work & money — trending now
    "This house believes that the return-to-office mandate is a power play, not a productivity strategy.",
    "This house believes that the four-day work week should become the global standard.",
    "This house believes that content creators have replaced journalists as the most influential media.",
    "This house believes that college degrees have become a luxury product most people cannot justify.",
    "This house believes that the gig economy exploits workers through algorithmic control.",
    # 🔥 Health & society — trending now
    "This house believes that GLP-1 drugs like Ozempic will end the obesity crisis.",
    "This house believes that smartphones have done more harm than good to teenagers.",
    "This house believes that lab-grown meat will replace conventional animal farming by 2040.",
    "This house believes that longevity research should be a top public-funding priority.",
    "This house believes that billionaires do more harm than good to democracy.",
    # 🔥 Environment & future — trending now
    "This house believes that nuclear energy is the only realistic path to net zero.",
    "This house believes that housing in major cities should be treated as a public utility.",
    "This house believes that humanity's future depends on becoming a multi-planetary species.",
    "This house believes that gene editing in humans should be fully legalized.",
    "This house believes that autonomous vehicles will make roads safer within a decade.",
]
