from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from workflow.states.candidate import compact_candidate
from workflow.states.answer_eval import AnswerEvaluation
from workflow.services.LLM import llm

prompt = ChatPromptTemplate.from_template("""
You are a strict Senior Technical Interviewer evaluating a candidate's answer.

Your ONLY job is to evaluate the latest answer. Do NOT generate questions.

Job Description:
{jd}

Candidate Resume:
{resume}

Question Asked:
{question}

Candidate Answer:
{answer}

Recent Conversation:
{history}

Scoring Rules (be strict):

- Technical Accuracy (1-10)
- Completeness (1-10)
- Relevance (1-10)
- Communication (1-10)
- Confidence (1-10)

Special handling rules (MUST follow):

1. If the answer is "I don't know", empty, pure filler, or completely unrelated:
   → technical_accuracy ≤ 2, completeness ≤ 2, relevance ≤ 3, confidence ≤ 3

2. If the answer is mostly unintelligible / transcription garbage:
   → technical_accuracy = 1, completeness = 1, communication ≤ 3

3. Never give high scores just because the candidate spoke confidently while saying almost nothing useful.

4. A weak answer is still valid evidence. Do not be generous.

Also provide:
- strengths (list)
- weaknesses (list)
- missing_information (list of specific things still needed)
- follow_up_required (true only if another attempt is genuinely useful)
- difficulty_recommendation: EASY | MEDIUM | HARD

Return ONLY valid JSON.

{format_instructions}
""")

parser = JsonOutputParser(pydantic_object=AnswerEvaluation)

chain = prompt | llm | parser


def evaluate_answer(state):
    question = state["history"][-1]["question"]
    answer = state["history"][-1]["answer"]
    history = state.get("history", [])[-2:]
    candidate_context = compact_candidate(state["candidate"])

    result = chain.invoke({
        "jd": state["jd"][:3000],
        "resume": candidate_context,
        "question": question[:1000],
        "answer": answer[:2500],
        "history": history,
        "format_instructions": parser.get_format_instructions()
    })

    return {
        "last_evaluation": result,
    }