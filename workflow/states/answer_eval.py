from pydantic import BaseModel, Field
from typing import List


class AnswerEvaluation(BaseModel):
    technical_accuracy: int = Field(description="Score 1-10")
    completeness: int = Field(description="Score 1-10")
    communication: int = Field(description="Score 1-10")
    confidence: int = Field(description="Score 1-10")
    relevance: int = Field(description="Score 1-10")

    strengths: List[str]
    weaknesses: List[str]
    missing_information: List[str]

    follow_up_required: bool

    difficulty_recommendation: str

    summary: str
    last_evaluation: str