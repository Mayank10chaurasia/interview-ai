from typing import TypedDict, Annotated, Optional, Dict, List
from operator import add

from workflow.states.candidate import CandidateProfile
from workflow.states.answer_eval import AnswerEvaluation
from workflow.states.decision import InterviewDecision


class QAPair(TypedDict):
    question: str
    answer: str


class InterviewState(TypedDict):
    candidate: CandidateProfile
    jd: str

    question: Optional[str]
    answer: Optional[str]

    history: Annotated[list[QAPair], add]
    evaluations: Annotated[list[AnswerEvaluation], add]

    last_evaluation: Optional[AnswerEvaluation]

    current_topic: str
    difficulty: str
    manager: Optional[InterviewDecision]

    question_count: int
    interview_completed: bool

    # ========== NEW FIELDS ==========
    topic_attempts: Dict[str, int]              # {"Python": 3, "MongoDB": 2}
    consecutive_weak_answers: int
    covered_topics: List[str]