from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from workflow.states.decision import InterviewDecision

from workflow.services.LLM import llm1

prompt = ChatPromptTemplate.from_template("""
You are an AI Interview Manager.

You NEVER ask interview questions.

Your only job is to decide what should happen next.

You receive:

Job Description
{jd}

Resume
{resume}

Conversation History
{history}

Latest Answer Evaluation
{evaluation}

Current Topic
{topic}

--------------------------------

Your decision must be ONE of:

FOLLOW_UP
- Candidate's answer is incomplete.
- Important evidence is still missing.

INCREASE_DIFFICULTY
- Candidate answered confidently.
- Challenge them with a harder question on the SAME topic.

NEXT_TOPIC
- Enough evidence has been collected.
- Choose the next most important topic from the resume or job description.

END_INTERVIEW
- All required competencies have enough evidence.

--------------------------------

Rules

Do NOT write interview questions.

Do NOT evaluate the answer.

Only decide:

1. What action?
2. Why?
3. Which topic?
4. Difficulty?
5. What evidence should the next question collect?

Return ONLY JSON.

{format_instructions}
""")

parser = JsonOutputParser(
    pydantic_object=InterviewDecision
)

chain = prompt | llm1 | parser

def interview_manager(state):
    history = state.get("history", [])[-5:]

    result = chain.invoke(
        {
            "jd": state["jd"],
            "resume": state["candidate"],
            "history": history,
            "evaluation": state.get("last_evaluation"),
            "topic": state.get("current_topic", "Introduction"),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    print("\nManager Decision:")
    print(result)

    return {
        "manager": result,
        "current_topic": result.get("next_topic")
            or result.get("current_topic", "Introduction"),
        "difficulty": result.get("next_difficulty", "EASY"),
    }