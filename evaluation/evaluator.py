"""Evaluation module for the Medical Appointment Agent."""

from dataclasses import dataclass, field


@dataclass
class AppointmentEvaluation:
    """Evaluation for a single patient interaction."""
    patient_name: str
    patient_age: int
    description_snippet: str
    detected_specialty: str
    final_specialty: str
    severity: str
    urgency_score: int
    sentiment: str
    doctor_assigned: str
    appointment_id: str
    time_slot: str
    status: str
    processing_steps: int
    processing_time: float
    had_error: bool


@dataclass
class AgentEvaluation:
    """Aggregate evaluation."""
    results: list[AppointmentEvaluation] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def booked_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.status == "confirmed") / self.total

    @property
    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.had_error) / self.total

    @property
    def avg_urgency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.urgency_score for r in self.results) / self.total

    @property
    def avg_time(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.processing_time for r in self.results) / self.total

    @property
    def specialty_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self.results:
            dist[r.final_specialty] = dist.get(r.final_specialty, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    @property
    def severity_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self.results:
            dist[r.severity] = dist.get(r.severity, 0) + 1
        return dist

    def summary_dict(self) -> dict:
        return {
            "total_patients": self.total,
            "booking_rate": f"{self.booked_rate:.0%}",
            "error_rate": f"{self.error_rate:.0%}",
            "avg_urgency": f"{self.avg_urgency:.1f}/10",
            "avg_processing_time": f"{self.avg_time:.1f}s",
            "specialties": self.specialty_distribution,
            "severities": self.severity_distribution,
        }


def evaluate_result(result: dict, elapsed: float) -> AppointmentEvaluation:
    return AppointmentEvaluation(
        patient_name=result.get("patient_name", ""),
        patient_age=result.get("patient_age", 0),
        description_snippet=result.get("description", "")[:60],
        detected_specialty=result.get("detected_specialty", ""),
        final_specialty=result.get("final_specialty", ""),
        severity=result.get("final_severity", ""),
        urgency_score=result.get("urgency_score", 0),
        sentiment=result.get("sentiment_label", ""),
        doctor_assigned=result.get("assigned_doctor_name", "N/A"),
        appointment_id=result.get("appointment_id", "N/A"),
        time_slot=result.get("time_slot", "N/A"),
        status=result.get("appointment_status", "unknown"),
        processing_steps=len(result.get("processing_log", [])),
        processing_time=elapsed,
        had_error=bool(result.get("error")),
    )
