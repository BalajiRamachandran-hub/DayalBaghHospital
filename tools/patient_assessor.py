"""Patient assessor — scores urgency based on age, symptoms, and sentiment."""


# Sentiment keywords
_DISTRESS_WORDS = {
    "severe": 3, "unbearable": 3, "extreme": 3, "worst": 3, "dying": 4,
    "going to die": 4, "die": 4, "death": 4,
    "emergency": 4, "critical": 4, "can't breathe": 4, "collapsed": 4,
    "terrible": 2, "horrible": 2, "very painful": 3, "excruciating": 3,
    "bleeding": 3, "blood loss": 4, "blood": 3, "unconscious": 4, "unresponsive": 4,
    "fracture": 3, "accident": 3, "trauma": 3, "broken": 3, "hit by": 3,
    "fall": 2, "injured": 2, "swollen": 2, "can't move": 3, "can't walk": 3,
    "paralyzed": 4, "numb": 2, "convulsion": 4, "seizure": 4, "choking": 4,
    "heart attack": 4, "chest pain": 3, "stroke": 4,
    "scared": 2, "worried": 1, "afraid": 2, "frightened": 2,
    "please help": 2, "urgent": 2, "immediately": 2, "immediate": 2, "asap": 2,
    "can't sleep": 1, "can't eat": 1, "days": 1, "weeks": 2, "months": 2,
    "getting worse": 2, "worsening": 2, "spreading": 1, "sudden": 2,
}

_CALM_WORDS = {
    "mild": -1, "slight": -1, "little": -1, "minor": -1,
    "routine": -2, "checkup": -2, "just wondering": -1,
    "occasional": -1, "sometimes": -1,
}


def assess_patient(description: str, age: int) -> dict:
    """Score patient urgency based on description sentiment and age.

    Returns:
        dict with: sentiment_score (-1 to 1), sentiment_label, urgency_score (1-10),
        age_risk_factor, matched_signals
    """
    text_lower = description.lower()
    distress = 0.0
    calm = 0.0
    signals: list[str] = []

    for word, weight in _DISTRESS_WORDS.items():
        if word in text_lower:
            distress += weight
            signals.append(f"distress:{word}")

    for word, weight in _CALM_WORDS.items():
        if word in text_lower:
            calm += abs(weight)
            signals.append(f"calm:{word}")

    # Sentiment score: -1 (very distressed) to +1 (calm/routine)
    total = distress + calm
    if total == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = (calm - distress) / total
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    if sentiment_score < -0.3:
        sentiment_label = "distressed"
    elif sentiment_score > 0.3:
        sentiment_label = "calm"
    else:
        sentiment_label = "concerned"

    # Age risk factor (0-3)
    if age < 5:
        age_risk = 3  # Infants/toddlers — high risk
    elif age < 12:
        age_risk = 2  # Children
    elif age <= 60:
        age_risk = 1  # Adults
    elif age <= 75:
        age_risk = 2  # Senior citizens
    else:
        age_risk = 3  # Elderly 75+

    # Urgency score (1-10)
    urgency = 3  # base
    urgency += min(6, distress / 1.5)  # distress adds up to 6
    urgency += age_risk * 0.5          # age adds up to 1.5
    urgency -= calm * 0.3             # calm reduces slightly
    urgency = max(1, min(10, round(urgency)))

    return {
        "sentiment_score": round(sentiment_score, 2),
        "sentiment_label": sentiment_label,
        "urgency_score": urgency,
        "age_risk_factor": age_risk,
        "matched_signals": signals,
    }
