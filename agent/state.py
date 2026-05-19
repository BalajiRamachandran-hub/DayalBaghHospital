"""Agent state definition for the Medical Appointment LangGraph workflow."""

from typing import Annotated
from typing_extensions import TypedDict


def _append_log(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


class AgentState(TypedDict):
    """State flowing through the appointment agent."""

    # Patient input
    patient_name: str
    patient_age: int
    description: str
    appointment_date: str  # YYYY-MM-DD

    # Symptom analysis
    detected_specialty: str
    severity_hint: str
    symptom_confidence: float
    matched_keywords: list[str]

    # Patient assessment
    sentiment_label: str
    sentiment_score: float
    urgency_score: int
    age_risk_factor: int

    # LLM classification (may refine specialty)
    llm_specialty: str
    llm_severity: str

    # Final specialty (resolved from tools + LLM)
    final_specialty: str
    final_severity: str

    # Doctor assignment
    assigned_doctor_id: str
    assigned_doctor_name: str
    assigned_doctor_qualification: str
    assigned_doctor_room: str

    # Appointment
    appointment_id: str
    time_slot: str
    appointment_status: str  # "confirmed", "no_slot", "no_doctor"

    # Response
    response_message: str

    # Tracing
    processing_log: Annotated[list[str], _append_log]
    current_node: str
    error: str
