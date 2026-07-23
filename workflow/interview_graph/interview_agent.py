from dotenv import load_dotenv
import os
from pprint import pprint

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from workflow.states.interview import InterviewState

from workflow.interview_graph.nodes.question_gen import generate_question
from workflow.interview_graph.nodes.tts import tts
from workflow.interview_graph.nodes.stt import speech_to_text
from workflow.interview_graph.nodes.answer import evaluate_answer
from workflow.interview_graph.nodes.interview_manager import interview_manager

load_dotenv()


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
# -----------------------------
# Build Graph
# -----------------------------
builder = StateGraph(InterviewState)


builder.add_node("generate_question", generate_question)
builder.add_node("tts", tts)
builder.add_node("speech_to_text", speech_to_text)
builder.add_node("answer_evaluator", evaluate_answer)
builder.add_node("interview_manager", interview_manager)


# -----------------------------
# Edges
# -----------------------------
builder.add_edge(START, "generate_question")

builder.add_edge("generate_question", "tts")
builder.add_edge("tts", "speech_to_text")
builder.add_edge("speech_to_text", "answer_evaluator")
builder.add_edge("answer_evaluator", "interview_manager")


builder.add_conditional_edges(
    "interview_manager",
    route_interview,
    {
        "continue": "generate_question",
        "end": END,
    },
)


# -----------------------------
# Checkpointer
# -----------------------------
DB_URI = os.getenv("CHECKPOINT_POSTGRES_URI")

config = {
    "configurable": {
        "thread_id": "thread-2"
    }
}


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    checkpointer.setup()

    graph = builder.compile(
        
        checkpointer=checkpointer
    )

    print("✅ Graph compiled successfully")

    existing = graph.get_state(config)

    if (
        existing.values
        and existing.values.get("candidate")
        and not existing.values.get("interview_completed")
    ):

        print("▶ Continuing interview...")

        result = graph.invoke(
            {
    "current_topic": "Introduction",
    "difficulty": "EASY",
    "manager": {
        "action": "FOLLOW_UP",
        "current_topic": "Introduction",
        "next_topic": None,
        "next_difficulty": "EASY",
        "reason": "",
        "objectives": [],
    },
    "last_evaluation": None,
    "evaluations": [],
},
            config=config,
        )

    elif (
        existing.values
        and existing.values.get("interview_completed")
    ):

        print("✅ Interview already completed.")
        result = existing.values

    else:

        raise RuntimeError(
            "Candidate/JD not found in checkpoint."
        )


checkpoint = graph.get_state(config)

print("\n========== FINAL STATE ==========")

pprint(checkpoint.values)