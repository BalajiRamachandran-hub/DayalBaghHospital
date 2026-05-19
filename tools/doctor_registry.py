"""Doctor registry — manages doctor database and availability lookup."""

import json
import os
from dataclasses import dataclass


@dataclass
class Doctor:
    id: str
    name: str
    specialty: str
    qualification: str
    experience_years: int
    room: str
    available_days: list[str]
    max_patients_per_day: int


class DoctorRegistry:
    """Loads and queries the doctor database."""

    def __init__(self, path: str | None = None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "doctors.json")
        with open(path) as f:
            raw = json.load(f)
        self.doctors = [
            Doctor(
                id=d["id"], name=d["name"], specialty=d["specialty"],
                qualification=d["qualification"], experience_years=d["experience_years"],
                room=d["room"], available_days=d["available_days"],
                max_patients_per_day=d["max_patients_per_day"],
            )
            for d in raw
        ]

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
