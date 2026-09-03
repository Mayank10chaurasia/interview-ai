from workflow.services.LLM import llm
llm.temperature = 0.35
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from workflow.states.candidate import compact_candidate


class NextQuestion(BaseModel):
    question: str = Field(description="Single next interview question")


parser = JsonOutputParser(pydantic_object=NextQuestion)

prompt = ChatPromptTemplate.from_template("""
You are conducting a realistic live technical interview.

Current stage/topic: {topic}
Difficulty: {difficulty}
Manager action: {action}
What the manager wants to learn: {objectives}

Candidate: {candidate}
Job: {jd}

Recent conversation:
{history}

HARD RULES (must follow):

- Ask exactly ONE question.
- Maximum 22 words.
- Sound natural and conversational (like a real interviewer).
- NEVER repeat or rephrase any previous question.
- If the previous answer was weak or "I don't know", ask a simpler or different angle — do not push the same point.
- If action is NEXT_TOPIC → cleanly transition to the new topic. Do not reference the old one.
- If action is FOLLOW_UP → dig into the specific missing information only if it is still useful.
- Prefer asking about real usage in the candidate’s projects rather than pure theory.

Return ONLY valid JSON.

{format_instructions}
""")

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)


def _format_history(history):
    if not history:
        return "No previous questions."
    lines = []
    for turn in history:
        lines.append(f"Q: {turn.get('question', '')}")
        if turn.get("answer"):
            lines.append(f"A: {turn.get('answer', '')}")
    return "\n".join(lines)


def generate_question(state):
    candidate_context = compact_candidate(state["candidate"])
    recent_history = state.get("history", [])[-4:]

    manager = state.get("manager", {})

    topic = state.get("current_topic", "Introduction")

    # FIRST QUESTION MUST ALWAYS BE INTRODUCTION
    if state.get("question_count", 0) == 0:
        return {
            "question": "Can you briefly introduce yourself and walk me through your background?",
            "question_count": 1,
        }

    result = chain.invoke({
        "jd": state["jd"][:2000],
        "candidate": candidate_context,
        "history": _format_history(recent_history),
        "topic": topic,
        "difficulty": state.get("difficulty", "EASY"),
        "action": manager.get("action", "FOLLOW_UP"),
        "objectives": manager.get("objectives", []),
        "format_instructions": parser.get_format_instructions(),
    })

    print("\nQUESTION GENERATOR OUTPUT:")
    print(result)

    return {
        "question": result["question"],
        "question_count": state.get("question_count", 0) + 1,
    }