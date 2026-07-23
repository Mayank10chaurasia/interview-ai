from typing import List, Optional
from pydantic import BaseModel, Field


class InterviewDecision(BaseModel):

    action: str = Field(
        description="FOLLOW_UP, INCREASE_DIFFICULTY, NEXT_TOPIC, END_INTERVIEW"
    )

    current_topic: str

    next_topic: Optional[str] = None

    next_difficulty: str = Field(
        description="EASY, MEDIUM or HARD"
    )

    reason: str

    objectives: List[str] = Field(
        description="Information the next question should obtain."
    )