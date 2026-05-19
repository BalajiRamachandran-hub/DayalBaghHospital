"""Symptom analyzer — keyword-based mapping of symptoms to specialties."""

import json
import os


class SymptomAnalyzer:
    """Maps patient-described symptoms to medical specialties using keyword matching."""

    def __init__(self, path: str | None = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "symptom_map.json")
        with open(path) as f:
            self.symptom_map: dict = json.load(f)

    def analyze(self, description: str) -> dict:
        """Analyze symptoms and return matched specialty + severity hint.

        Returns:
            dict with: specialty, severity_hint, matched_keywords, confidence
        """
        desc_lower = description.lower()
        scores: dict[str, dict] = {}

        for category, info in self.symptom_map.items(): 
            matched = []
            for kw in info["keywords"]:
                if kw in desc_lower:
                    matched.append(kw)
            if matched:
                scores[category] = {
                    "specialty": info["specialty"],
                    "severity_hint": info["severity_hint"],
                    "matched_keywords": matched,
                    "score": len(matched),
                }

        if not scores:
            return {
                "specialty": "general_medicine",
                "severity_hint": "low",
                "matched_keywords": [],
                "confidence": 0.0,
                "all_matches": [],
            }

        # Pick best match by score
        best = max(scores.values(), key=lambda x: x["score"])
        total_possible = max(len(info["keywords"]) for info in self.symptom_map.values())
        confidence = min(1.0, best["score"] / 3)  # 3+ keyword matches = full confidence

        all_matches = [
            {"specialty": v["specialty"], "keywords": v["matched_keywords"], "score": v["score"]}
            for v in sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        ]

        return {
            "specialty": best["specialty"],
            "severity_hint": best["severity_hint"],
            "matched_keywords": best["matched_keywords"],
            "confidence": round(confidence, 2),
            "all_matches": all_matches,
        }
