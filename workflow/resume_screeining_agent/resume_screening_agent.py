from langsmith import traceable
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from workflow.resume_screeining_agent.nodes.email_genrator import email_agent
import os
from workflow.states.candidate import CandidateProfile
from langgraph.checkpoint.postgres import PostgresSaver
from workflow.services.LLM import llm




class ResumeState(StateGraph): 
    pdf_path: str 
    jd: str 
    resume_text: str 
    chunks: list 
    retrieved_context: str 
    score: int 
    analysis: dict
    decision: str
    candidate: CandidateProfile


load_dotenv()

from pydantic import BaseModel

class ResumeAnalysis(BaseModel):
    overall_match: float
    strengths: list[str]
    weaknesses: list[str]
    missing_skills: list[str]

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(
    pydantic_object=ResumeAnalysis
)

from langchain_core.prompts import ChatPromptTemplate

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
@traceable(name="analyzer")
def analyze_resume(state):

    result = chain.invoke({
        "jd": state["jd"],
        "resume": state["retrieved_context"]
    })

    state["analysis"] = result

    return state



@traceable(name="score")
def score_resume(state):

    

    state["score"] = state["analysis"]["overall_match"]


    return state

@traceable(name="decision_making")
def decision(state):

    if state["score"] >= 60:
        state["decision"] = "interview"

    else:
        state["decision"] = "reject"

    return state






#building graph 



from workflow.resume_screeining_agent.nodes.load_chunk_emdd import (
    load_resume,
    chunk_resume,
    retrieve_resume,
    extract_candidate
)

builder = StateGraph(ResumeState)

builder.add_node("load_resume", load_resume)

builder.add_node("chunk_resume", chunk_resume)

builder.add_node("extract_candidate", extract_candidate)

builder.add_node("retrieve_resume", retrieve_resume)

builder.add_node("analysis", analyze_resume)

builder.add_node("score", score_resume)

builder.add_node("decision", decision)
builder.add_node("email_gen", email_agent)

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

builder.add_edge("email_gen", END)




state = {
    "pdf_path":"TCET_Resume (1).pdf",

    "jd": """
AI Engineer Intern

Skills

Python
Mongodb
react
Machine Learning

"""
}

DB_URI = os.getenv("CHECKPOINT_POSTGRES_URI")
config = {
    "configurable": {
        "thread_id": "thread-2"
    }}

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(state,config=config)
    png = graph.get_graph().draw_mermaid_png()
    checkpoint = graph.get_state(config)
    with open("graph.png", "wb") as f:
        f.write(png)

print("Saved as graph.png")

from pprint import pprint

print("========== CURRENT STATE ==========")
pprint(checkpoint.values)