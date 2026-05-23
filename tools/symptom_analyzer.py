"""Symptom analyzer — NER + NLTK-based extraction with keyword matching."""

import json
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (silent if already present)
for resource in ["punkt_tab", "averaged_perceptron_tagger_eng", "maxent_ne_chunker_tab", "words", "wordnet"]:
    nltk.download(resource, quiet=True)


class SymptomAnalyzer:
    """Maps patient-described symptoms to medical specialties using NER + NLTK noun extraction + keyword matching."""

    def __init__(self, path: str | None = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "symptom_map.json")
        with open(path) as f:
            self.symptom_map: dict = json.load(f)
        self._lemmatizer = WordNetLemmatizer()

    def _extract_medical_terms(self, description: str) -> list[str]:
        """Use NLTK to extract nouns, noun phrases, and named entities from the description."""
        tokens = word_tokenize(description.lower())
        tagged = pos_tag(tokens)

        # Extract nouns (NN, NNS, NNP, NNPS) and adjectives describing conditions (JJ)
        nouns = []
        for word, tag in tagged:
            if tag in ("NN", "NNS", "NNP", "NNPS"):
                lemma = self._lemmatizer.lemmatize(word)
                nouns.append(lemma)
            elif tag == "JJ":
                # Adjectives like "severe", "burning", "blurry" are medically relevant
                nouns.append(word)

        # Extract named entities using NLTK NER
        tree = ne_chunk(pos_tag(word_tokenize(description)))
        entities = []
        for subtree in tree:
            if hasattr(subtree, "label"):
                entity = " ".join(word for word, tag in subtree.leaves())
                entities.append(entity.lower())

        # Build bigrams and trigrams for multi-word symptom matching
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]

        # Combine all extracted terms
        all_terms = nouns + entities + bigrams + trigrams
        return list(set(all_terms))

    def analyze(self, description: str) -> dict:
        """Analyze symptoms using NER + noun extraction + keyword matching.

        Returns:
            dict with: specialty, severity_hint, matched_keywords, confidence,
                       extracted_terms, all_matches
        """
        desc_lower = description.lower()
        extracted_terms = self._extract_medical_terms(description)

        scores: dict[str, dict] = {}

        for category, info in self.symptom_map.items():
            matched = []
            for kw in info["keywords"]:
                # Match against full description (substring match)
                if kw in desc_lower:
                    matched.append(kw)
                # Match against NER-extracted terms (require meaningful overlap)
                elif any(
                    (kw in term and len(kw) >= 4) or
                    (term in kw and len(term) >= 5 and len(term) >= len(kw) * 0.8)
                    for term in extracted_terms
                ):
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
                "extracted_terms": extracted_terms,
                "all_matches": [],
            }

        # Pick best match by score
        best = max(scores.values(), key=lambda x: x["score"])
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
            "extracted_terms": extracted_terms,
            "all_matches": all_matches,
        }
