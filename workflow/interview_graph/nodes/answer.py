from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import JsonOutputParser

from workflow.states.answer_eval import AnswerEvaluation
from workflow.services.LLM import llm1
prompt = ChatPromptTemplate.from_template("""
You are an experienced Senior Technical Interviewer.

Your ONLY responsibility is to evaluate the candidate's latest answer.

Do NOT generate another interview question.

Job Description:
{jd}

Resume:
{resume}

Question Asked:
{question}

Candidate Answer:
{answer}

Conversation History:
{history}

Evaluate the answer using these criteria:

1. Technical Accuracy (1-10)
2. Completeness (1-10)
3. Relevance (1-10)
4. Communication (1-10)
5. Confidence (1-10)

Also identify:

- strengths
- weaknesses
- missing_information

If important information is missing,
set follow_up_required = true.

Recommend one difficulty level:

EASY
MEDIUM
HARD

Return ONLY JSON.

{format_instructions}
""")




parser = JsonOutputParser(pydantic_object=AnswerEvaluation)

chain = (
    prompt
    | llm1
    | parser
)


def evaluate_answer(state):

    question = state["history"][-1]["question"]
    answer = state["history"][-1]["answer"]
    history = state.get("history", [])[-5:]

    result = chain.invoke(
        {
            "jd": state["jd"],
            "resume": state["candidate"],
            "question": question,
            "answer": answer,
            "history": history,
            "format_instructions": parser.get_format_instructions()
        }
    )

    
    
    return {
    "last_evaluation": result,
}