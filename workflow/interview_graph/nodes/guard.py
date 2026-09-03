def force_decision(state):
    """
    Hard rules that override the LLM Manager when necessary.
    """
    manager = state.get("manager") or {}
    action = manager.get("action")

    topic = state.get("current_topic", "")
    attempts = (state.get("topic_attempts") or {}).get(topic, 0)
    weak = state.get("consecutive_weak_answers", 0)
    total = state.get("question_count", 0)
    covered = state.get("covered_topics") or []

    # ---------- HARD RULES ----------
    # 1. Too many weak answers overall → END
    if weak >= 4 or total >= 14:
        return {
            "manager": {
                **manager,
                "action": "END_INTERVIEW",
                "reason": "Too many weak/evasive answers or max questions reached"
            }
        }

    # 2. Current topic already attempted too many times → force NEXT_TOPIC
    if attempts >= 3:
        return {
            "manager": {
                **manager,
                "action": "NEXT_TOPIC",
                "reason": f"Forced: {topic} already attempted {attempts} times"
            }
        }

    # 3. If topic is already covered and Manager wants to stay → force move
    if topic in covered and action == "FOLLOW_UP":
        return {
            "manager": {
                **manager,
                "action": "NEXT_TOPIC",
                "reason": f"Forced: {topic} already covered"
            }
        }

    return {}