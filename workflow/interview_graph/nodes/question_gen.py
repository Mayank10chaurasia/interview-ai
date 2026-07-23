from workflow.services.LLM import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class NextQuestion(BaseModel):
    question: str = Field(description="The single next interview question, conversational tone")

parser = JsonOutputParser(pydantic_object=NextQuestion)

prompt = ChatPromptTemplate.from_template("""
You are a Senior Engineering Manager conducting a realistic live interview.

Use the candidate's resume, job description, and conversation history to ask the next best interview question.

## Job Description
{jd}

## Candidate Resume
{candidate}

## Conversation History
{history}

### Interview Flow
1. Introduction - Welcome the candidate, discuss background, education/work. No technical questions.
2. Resume - Discuss academics, career choices, internships, strengths, weaknesses, and achievements.
3. Projects -Start with the latest project, then explore motivation, architecture, challenges, decisions, and alternatives.
4. Technical - Begin with basic concepts, gradually increase difficulty, and ask follow-up questions based on previous answers.
5. Problem Solving -Ask debugging, scenario-based, or system design questions.
6. Behavioral -Explore teamwork, leadership, communication, conflict resolution, and deadlines.
7. Closing -Ask if the candidate has any questions.

### Rules
- Ask exactly ONE question.
- Never ask multiple or repeated questions.
- Continue naturally from the previous answer.
- Probe deeper if the answer is weak; increase difficulty if it is strong.
- Complete the current stage before moving to the next.
- Never jump from introduction directly to projects or technical questions.
- For the first question, welcome the candidate and ask about their journey.
- Keep the question conversational and under 25 words.

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
        return "(no questions asked yet)"
    lines = []
    for turn in history:
        lines.append(f"Q: {turn['question']}")
        if turn.get("answer"):
            lines.append(f"A: {turn['answer']}")
    return "\n".join(lines)


def generate_question(state):
    
    result = chain.invoke({
        "jd": state.get("jd", """AI Engineer Intern

Skills

Python
Mongodb
react
Machine Learning"""),
        "candidate": state["candidate"],
        "history": _format_history(state.get("history", [])),
    })
    print(result)

    return {
        "question": result["question"],          # unwrap, don't re-nest
        "question_count": state.get("question_count", 0) + 1,
    }