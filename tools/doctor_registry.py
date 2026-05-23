"""Doctor registry — manages doctor database with cosine similarity matching."""

import json
import os
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Doctor:
    id: str
    name: str
    specialty: str
    treats: list[str]
    qualification: str
    experience_years: int
    room: str
    available_days: list[str]
    max_patients_per_day: int


# Minimum cosine similarity threshold to consider a match valid
SIMILARITY_THRESHOLD = 0.15


class DoctorRegistry:
    """Loads and queries the doctor database with cosine similarity matching."""

    def __init__(self, path: str | None = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "doctors.json")
        with open(path) as f:
            raw = json.load(f)
        self.doctors = [
            Doctor(
                id=d["id"], name=d["name"], specialty=d["specialty"],
                treats=d.get("treats", []),
                qualification=d["qualification"], experience_years=d["experience_years"],
                room=d["room"], available_days=d["available_days"],
                max_patients_per_day=d["max_patients_per_day"],
            )
            for d in raw
        ]
        # Pre-build TF-IDF vectors for each doctor's treats list
        self._treats_corpus = [" ".join(d.treats) for d in self.doctors]
        self._vectorizer = TfidfVectorizer()
        self._doctor_vectors = self._vectorizer.fit_transform(self._treats_corpus)

    def find_by_similarity(self, extracted_terms: list[str], day_name: str | None = None) -> list[tuple[Doctor, float]]:
        """Find doctors whose treats list is most similar to the extracted symptom terms.

        Returns a list of (Doctor, similarity_score) tuples sorted by score descending.
        Only returns doctors above SIMILARITY_THRESHOLD.
        """
        if not extracted_terms:
            return []

        # Build query from extracted terms
        query_text = " ".join(extracted_terms)
        query_vector = self._vectorizer.transform([query_text])

        # Compute cosine similarity against all doctors
        similarities = cosine_similarity(query_vector, self._doctor_vectors)[0]

        # Pair with doctors and filter
        results = []
        for doc, score in zip(self.doctors, similarities):
            if score >= SIMILARITY_THRESHOLD:
                if day_name is None or day_name in doc.available_days:
                    results.append((doc, round(float(score), 4)))

        # Sort by similarity score (highest first), then by experience
        results.sort(key=lambda x: (-x[1], -x[0].experience_years))
        return results

    def find_by_specialty(self, specialty: str, day_name: str | None = None) -> list[Doctor]:
        """Find doctors matching a specialty, optionally filtering by available day."""
        results = [d for d in self.doctors if d.specialty == specialty]
        if day_name:
            results = [d for d in results if day_name in d.available_days]
        # Sort by experience (most experienced first)
        results.sort(key=lambda d: d.experience_years, reverse=True)
        return results

    def get_doctor(self, doctor_id: str) -> Doctor | None:
        for d in self.doctors:
            if d.id == doctor_id:
                return d
        return None

    def get_all_specialties(self) -> list[str]:
        return sorted(set(d.specialty for d in self.doctors))
