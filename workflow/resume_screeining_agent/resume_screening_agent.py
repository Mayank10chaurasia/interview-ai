import os
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import StateGraph, END

from workflow.states.candidate import CandidateProfile
from workflow.services.LLM import llm

from workflow.resume_screeining_agent.nodes.email_genrator import (
    email_agent
)

from workflow.resume_screeining_agent.nodes.load_chunk_emdd import (
    load_resume,
    chunk_resume,
    retrieve_resume,
    extract_candidate,
)

from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# =========================================
# STATE
# =========================================

class ResumeState(TypedDict, total=False):
    pdf_path:str
    jd: str 
    resume_text: str
    chunks: list
    retrieved_context: str

    score: int
    analysis: dict
    decision: str

    candidate: CandidateProfile

    email_subject: str
    email_body: str
    email_status: str


# =========================================
# ANALYSIS OUTPUT
# =========================================

class ResumeAnalysis(BaseModel):
    overall_match: float
    strengths: list[str]
    weaknesses: list[str]
    missing_skills: list[str]


parser = JsonOutputParser(
    pydantic_object=ResumeAnalysis
)


# =========================================
# ANALYSIS PROMPT
# =========================================

prompt = ChatPromptTemplate.from_template("""
Compare the Resume with the Job Description.

Job Description:
{jd}

Resume:
{resume}

Return overall_match as an integer percentage between 0 and 100.

Return ONLY valid JSON.

{format_instructions}
""")


chain = (
    prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    | llm
    | parser
)


# =========================================
# ANALYZE NODE
# =========================================

@traceable(name="analyzer")
def analyze_resume(state):

    result = chain.invoke({
        "jd": state["jd"][:5000],
        "resume": state["retrieved_context"][:4000]
    })

    state["analysis"] = result

    return state

# =========================================
# SCORE NODE
# =========================================

@traceable(name="score")
def score_resume(state):

    score = int(
        state["analysis"]["overall_match"]
    )

    return {
        "score": score
    }


# =========================================
# DECISION NODE
# =========================================

@traceable(name="decision_making")
def decision(state):

    if state["score"] >= 60:
        decision_result = "interview"
    else:
        decision_result = "reject"

    return {
        "decision": decision_result
    }


# =========================================
# BUILD GRAPH
# =========================================

builder = StateGraph(ResumeState)

builder.add_node(
    "load_resume",
    load_resume
)

builder.add_node(
    "chunk_resume",
    chunk_resume
)

builder.add_node(
    "extract_candidate",
    extract_candidate
)

builder.add_node(
    "retrieve_resume",
    retrieve_resume
)

builder.add_node(
    "analysis",
    analyze_resume
)

builder.add_node(
    "score",
    score_resume
)

builder.add_node(
    "decision",
    decision
)

builder.add_node(
    "email_gen",
    email_agent
)


# =========================================
# EDGES
# =========================================

builder.set_entry_point("load_resume")

builder.add_edge(
    "load_resume",
    "chunk_resume"
)

builder.add_edge(
    "chunk_resume",
    "extract_candidate"
)

builder.add_edge(
    "extract_candidate",
    "retrieve_resume"
)

builder.add_edge(
    "retrieve_resume",
    "analysis"
)

builder.add_edge(
    "analysis",
    "score"
)

builder.add_edge(
    "score",
    "decision"
)


builder.add_conditional_edges(
    "decision",
    lambda state: state["decision"],
    {
        "interview": "email_gen",
        "reject": END,
    }
)


builder.add_edge(
    "email_gen",
    END
)


# =========================================
# COMPILE GRAPH
# =========================================

graph = builder.compile()


# =========================================
# FUNCTION FASTAPI WILL CALL
# =========================================

DB_URI = os.getenv("CHECKPOINT_POSTGRES_URI")


def run_resume_graph(state, application_id):

    config = {
        "configurable": {
            "thread_id": str(application_id)
        }
    }

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

        # Only really needed when initializing/migrating checkpoint tables
        checkpointer.setup()

        graph = builder.compile(
            checkpointer=checkpointer
        )

        result = graph.invoke(
            state,
            config=config
        )
        print("\n========== RESUME GRAPH RESULT ==========")
        print(result)
        print("=========================================\n")

        return result

    
