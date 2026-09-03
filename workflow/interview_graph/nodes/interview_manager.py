from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from workflow.states.candidate import compact_candidate
from workflow.states.decision import InterviewDecision
from workflow.services.LLM import llm

parser = JsonOutputParser(pydantic_object=InterviewDecision)

prompt = ChatPromptTemplate.from_template("""
You are a strict Interview Manager for a technical interview.

Your only job is to decide the next action.

### Hard Rules (must follow in priority order):

1. If consecutive_weak_answers >= 3 → choose END_INTERVIEW
2. If the current topic has already been attempted 3 or more times → choose NEXT_TOPIC
3. If the current topic is already in covered_topics → choose NEXT_TOPIC
4. If technical_accuracy ≤ 2 and completeness ≤ 2 → strongly prefer NEXT_TOPIC over FOLLOW_UP
5. Only use FOLLOW_UP when the previous answer was reasonably relevant and something specific is still missing
6. Prefer covering all important JD skills  rather than deep-diving one weak area
7. Use INCREASE_DIFFICULTY only if the last answer scored ≥ 7 on technical_accuracy
8. Use END_INTERVIEW when enough evidence has been collected or the candidate is clearly not performing

### Current State:
- Current topic: {topic}
- Topic attempts so far: {topic_attempts}
- Consecutive weak answers: {consecutive_weak}
- Already covered topics: {covered_topics}
- Latest evaluation: {evaluation}
- Recent conversation: {history}
- Candidate: {resume}
- Job Description: {jd}

Return ONLY valid JSON.

{format_instructions}
""")

chain = prompt | llm | parser


def interview_manager(state):
    candidate_context = compact_candidate(state.get("candidate", {}))

    raw_history = state.get("history", [])[-3:]
    history = []
    for item in raw_history:
        history.append({
            "question": str(item.get("question", ""))[:500],
            "answer": str(item.get("answer", ""))[:800],
        })

    evaluation = state.get("last_evaluation") or {}
    compact_evaluation = {
        "technical_accuracy": evaluation.get("technical_accuracy"),
        "completeness": evaluation.get("completeness"),
        "relevance": evaluation.get("relevance"),
        "communication": evaluation.get("communication"),
        "confidence": evaluation.get("confidence"),
        "missing_information": evaluation.get("missing_information", [])[:3],
        "follow_up_required": evaluation.get("follow_up_required"),
        "difficulty_recommendation": evaluation.get("difficulty_recommendation"),
    }

    result = chain.invoke({
        "jd": str(state.get("jd", ""))[:2500],
        "resume": candidate_context,
        "history": history,
        "evaluation": compact_evaluation,
        "topic": state.get("current_topic", "Introduction"),
        "topic_attempts": state.get("topic_attempts", {}),
        "consecutive_weak": state.get("consecutive_weak_answers", 0),
        "covered_topics": state.get("covered_topics", []),
        "format_instructions": parser.get_format_instructions(),
    })

    print("\n========== MANAGER ==========")
    print("Action:", result.get("action"))
    print("Topic:", result.get("next_topic") or result.get("current_topic"))
    print("Difficulty:", result.get("next_difficulty"))
    print("Reason:", result.get("reason"))
    print("=============================\n")

    return {
        "manager": result,
        "current_topic": result.get("next_topic") or result.get("current_topic") or state.get("current_topic", "Introduction"),
        "difficulty": result.get("next_difficulty", state.get("difficulty", "EASY")),
    }