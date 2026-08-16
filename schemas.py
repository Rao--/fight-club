"""
Data model. Pydantic for judge output (needs validation); dataclasses for the rest.
No Streamlit imports.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    exchange_num: int       # 1-5; closings use 0
    fighter: str            # "A" or "B"
    model: str              # ollama tag
    persona_name: str
    content: str
    seconds: float          # generation wall time
    is_closing: bool = False
    tokens_out: int = 0    # output tokens (Ollama eval_count)
    tokens_in: int = 0     # prompt tokens (Ollama prompt_eval_count)
    tps: float = 0.0       # tokens/second (eval_count / eval_duration)
    ttft: float = 0.0      # time to first token (seconds)


# ---------------------------------------------------------------------------
# Judge output
# ---------------------------------------------------------------------------

class TurnScore(BaseModel):
    fighter: Literal["A", "B"]
    logic: int              # 1-10
    evidence: int           # 1-10
    rhetoric: int           # 1-10
    responsiveness: int     # 1-10
    justification: str      # one sentence

    @property
    def total(self) -> int:
        return self.logic + self.evidence + self.rhetoric + self.responsiveness


class BlockVerdict(BaseModel):
    block_num: int
    covers_exchanges: list[int]
    scores: list[TurnScore]     # exactly 2, one per fighter
    block_winner: Literal["A", "B", "tie"]
    key_moment: str             # one sentence


def fallback_verdict(block_num: int, covers_exchanges: list[int]) -> BlockVerdict:
    """Neutral fallback when judge output cannot be parsed twice in a row."""
    return BlockVerdict(
        block_num=block_num,
        covers_exchanges=covers_exchanges,
        scores=[
            TurnScore(fighter="A", logic=5, evidence=5, rhetoric=5, responsiveness=5,
                      justification="Judge output unparseable."),
            TurnScore(fighter="B", logic=5, evidence=5, rhetoric=5, responsiveness=5,
                      justification="Judge output unparseable."),
        ],
        block_winner="tie",
        key_moment="Judge output unparseable.",
    )
