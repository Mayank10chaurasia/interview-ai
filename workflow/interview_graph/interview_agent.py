from dotenv import load_dotenv
import os

from langgraph.graph import (
    StateGraph,
    START,
    END,
)



from langgraph.types import interrupt

from langgraph.checkpoint.postgres import PostgresSaver

from workflow.states.interview import InterviewState

from workflow.interview_graph.nodes.question_gen import (
    generate_question,
)

from workflow.interview_graph.nodes.answer import (
    evaluate_answer,
)

from workflow.interview_graph.nodes.interview_manager import (
    interview_manager,
)


from workflow.interview_graph.nodes.tracking import update_tracking
from workflow.interview_graph.nodes.guard import force_decision

load_dotenv()




DB_URI = os.getenv("CHECKPOINT_POSTGRES_URI")


def wait_for_answer(state: InterviewState):

    # LangGraph pauses here.
    # FastAPI will return this question to React.
    answer = interrupt({
        "question": state["question"],
        "question_count": state.get(
            "question_count",
            1
        ),
    })

    # When FastAPI resumes the graph,
    # Command(resume=transcript) becomes `answer`.

    return {
        "answer": answer,

        "history": [
            {
                "question": state["question"],
                "answer": answer,
            }
        ],
    }
# -----------------------------
# Routing Function
# -----------------------------
def route_interview(state: InterviewState):
    decision = state.get("manager")

    if not decision:
        return "continue"

    if isinstance(decision, dict):
        action = decision.get("action")
    else:
        action = decision.action

    if action == "END_INTERVIEW":
        return "end"

    return "continue"


def complete_interview(state: InterviewState):

    return {
        "interview_completed": True
    }

builder = StateGraph(InterviewState)

builder.add_node("generate_question", generate_question)
builder.add_node("wait_for_answer", wait_for_answer)
builder.add_node("answer_evaluator", evaluate_answer)
builder.add_node("update_tracking", update_tracking)      # NEW
builder.add_node("interview_manager", interview_manager)
builder.add_node("force_decision", force_decision)        # NEW
builder.add_node("complete_interview", complete_interview)

# Edges
builder.add_edge(START, "generate_question")
builder.add_edge("generate_question", "wait_for_answer")
builder.add_edge("wait_for_answer", "answer_evaluator")
builder.add_edge("answer_evaluator", "update_tracking")   # NEW
builder.add_edge("update_tracking", "interview_manager")
builder.add_edge("interview_manager", "force_decision")   # NEW

builder.add_conditional_edges(
    "force_decision",
    route_interview,
    {
        "continue": "generate_question",
        "end": "complete_interview",
    },
)

builder.add_edge("complete_interview", END)
# -----------------------------
# Checkpointer
# -----------------------------

from langgraph.types import Command


# ==========================================
# START INTERVIEW
# ==========================================

def start_interview(application_id: str):

    print("APPLICATION:", application_id)

    print(
        "POSTGRES URI EXISTS:",
        bool(DB_URI)
    )

    config = {
        "configurable": {
            "thread_id": str(application_id)
        }
    }

    print("Connecting to Postgres...")

    with PostgresSaver.from_conn_string(
        DB_URI
    ) as checkpointer:

        print("Postgres connected!")

        checkpointer.setup()

        print("Checkpoint setup complete")

        graph = builder.compile(
            checkpointer=checkpointer
        )
        print("Graph compiled")

        # ==================================
        # LOAD EXISTING RESUME CHECKPOINT
        # ==================================

        existing = graph.get_state(config)
        print("Checkpoint loaded")
        print("Existing values:", existing.values)

        if not existing.values:
            raise RuntimeError(
                "Checkpoint not found."
            )

        if not existing.values.get("candidate"):
            raise RuntimeError(
                "Candidate not found in checkpoint."
            )

        if not existing.values.get("jd"):
            raise RuntimeError(
                "Job description not found in checkpoint."
            )

        if existing.values.get(
            "interview_completed"
        ):
            return {
                "completed": True,
                "message":
                    "Interview already completed.",
                "result": build_interview_result(existing.values),
            }

        # ==================================
        # START GRAPH
        # ==================================

        result = graph.invoke(
    {
        "current_topic": "Introduction",
        "difficulty": "EASY",
        "question_count": 0,
        "interview_completed": False,
        "last_evaluation": None,
        "topic_attempts": {},
        "consecutive_weak_answers": 0,
        "covered_topics": [],
        "manager": {
            "action": "FOLLOW_UP",
            "current_topic": "Introduction",
            "next_topic": None,
            "next_difficulty": "EASY",
            "reason": "",
            "objectives": [],
        },
    },
    config=config,
)

        # ==================================
        # GET CURRENT STATE
        # ==================================

        state = graph.get_state(config)

        if state.values.get("question") and state.next:
         return {
        "completed": False,
        "question": state.values.get("question"),
        "question_count": state.values.get(
            "question_count",
            0
        ),
    }
        


# ==========================================
# SUBMIT ANSWER
# ==========================================

def submit_answer(
    application_id: str,
    transcript: str,
):

    config = {
        "configurable": {
            "thread_id": str(application_id)
        }
    }

    with PostgresSaver.from_conn_string(
        DB_URI
    ) as checkpointer:

        graph = builder.compile(
            checkpointer=checkpointer
        )

        existing = graph.get_state(config)

        if not existing.values:
            raise RuntimeError(
                "Interview not found."
            )

        if existing.values.get(
            "interview_completed"
        ):
            return {
                "completed": True,
                "result": build_interview_result(existing.values),
            }
        result = graph.invoke(
            Command(
                resume=transcript
            ),

            config=config,
        )
        print(result)

        state = graph.get_state(config)

        # ==================================
        # INTERVIEW FINISHED
        # ==================================

        if state.values.get(
            "interview_completed"
        ):

            return {
                "completed": True,
                "question": None,
                "question_count":
                    state.values.get(
                        "question_count",
                        0
                    ),
                "result": build_interview_result(state.values),
            }

        # ==================================
        # NEXT QUESTION
        # ==================================

        return {
            "completed": False,

            "question":
                state.values.get("question"),

            "question_count":
                state.values.get(
                    "question_count",
                    0
                ),
        }


def build_interview_result(state):
    evaluations = state.get("evaluations") or []

    def score(name):
        values = [
            evaluation.get(name, 0)
            if isinstance(evaluation, dict)
            else getattr(evaluation, name, 0)
            for evaluation in evaluations
        ]
        return round(sum(values) / len(values) * 10) if values else 0

    strengths = []
    weaknesses = []
    for evaluation in evaluations:
        if isinstance(evaluation, dict):
            strengths.extend(evaluation.get("strengths") or [])
            weaknesses.extend(evaluation.get("weaknesses") or [])
            weaknesses.extend(evaluation.get("missing_information") or [])
        else:
            strengths.extend(evaluation.strengths or [])
            weaknesses.extend(evaluation.weaknesses or [])
            weaknesses.extend(evaluation.missing_information or [])

    unique = lambda items: list(dict.fromkeys(item for item in items if item))
    completeness = score("completeness")
    relevance = score("relevance")
    category_scores = [
        score("technical_accuracy"),
        score("communication"),
        score("confidence"),
        completeness,
        relevance,
    ]

    return {
        "overall": round(sum(category_scores) / len(category_scores)),
        "technical": category_scores[0],
        "communication": category_scores[1],
        "confidence": category_scores[2],
        "problemSolving": round((completeness + relevance) / 2),
        "strengths": unique(strengths),
        "areasToImprove": unique(weaknesses),
        "questionCount": state.get("question_count", 0),
    }