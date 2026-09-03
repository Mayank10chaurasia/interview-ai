def update_tracking(state):
    topic = state.get("current_topic", "Unknown")
    evaluation = state.get("last_evaluation") or {}

    # Update topic attempts
    topic_attempts = dict(state.get("topic_attempts") or {})
    topic_attempts[topic] = topic_attempts.get(topic, 0) + 1

    # Detect weak answer
    tech = evaluation.get("technical_accuracy", 10)
    complete = evaluation.get("completeness", 10)
    relevance = evaluation.get("relevance", 10)

    is_weak = tech <= 2 or complete <= 2 or relevance <= 3

    consecutive = state.get("consecutive_weak_answers", 0)
    consecutive = consecutive + 1 if is_weak else 0

    # Mark topic as covered if it has been attempted enough times
    covered = list(state.get("covered_topics") or [])
    if topic_attempts[topic] >= 2 and topic not in covered:
        covered.append(topic)

    return {
        "topic_attempts": topic_attempts,
        "consecutive_weak_answers": consecutive,
        "covered_topics": covered,
        "evaluations": [evaluation] if evaluation else [],
    }