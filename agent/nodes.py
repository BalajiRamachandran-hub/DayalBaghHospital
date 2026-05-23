"""Graph nodes — each function is one step in the appointment agent workflow."""

import re
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from config import (
    HOSPITAL_NAME,
    HOSPITAL_PHONE,
    MAX_RETRIES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SPECIALTIES,
    SEVERITY_LEVELS,
)
from tools.doctor_registry import DoctorRegistry
from tools.symptom_analyzer import SymptomAnalyzer
from tools.patient_assessor import assess_patient
from tools.appointment_manager import AppointmentManager

# Shared resources
_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
_symptom_analyzer = SymptomAnalyzer()
_doctor_registry = DoctorRegistry()
_appt_manager = AppointmentManager()


def get_appointment_manager() -> AppointmentManager:
    """Expose the shared appointment manager for main.py."""
    return _appt_manager


def _ask_llm(prompt: str) -> str:
    response = _llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# PROMPT STRATEGIES — zero-shot, one-shot, chain-of-thought
# ---------------------------------------------------------------------------

class PromptStrategy:
    """Wraps a base prompt with different prompting techniques."""

    @staticmethod
    def zero_shot(base_prompt: str) -> str:
        """Direct instruction, no examples. Fast but may be less accurate for edge cases."""
        return base_prompt

    @staticmethod
    def one_shot(base_prompt: str) -> str:
        """Provides one complete example before the actual task."""
        return f"""Here is an example of how to perform medical triage classification:

--- EXAMPLE ---
You are a medical triage assistant at Dayal Bagh Hospital.

PATIENT AGE: 62 years
SYMPTOMS DESCRIBED: I have been feeling heaviness in my chest since last night. There is pain radiating to my left arm and I am sweating a lot. I also feel nauseous.
KEYWORD MATCH SUGGESTED: cardiology

SPECIALTY: cardiology
SEVERITY: critical
--- END EXAMPLE ---

Now classify the following patient using the same format:

{base_prompt}"""

    @staticmethod
    def chain_of_thought(base_prompt: str) -> str:
        """Asks LLM to reason step-by-step before giving final answer."""
        return f"""{base_prompt}

Before giving your final answer, reason through these steps:
Step 1 - Body System: Which organ system or body part is primarily affected based on the symptoms described?
Step 2 - Likely Condition: What medical condition do these symptoms most likely indicate?
Step 3 - Urgency Assessment: Given the patient's age and symptom severity, how urgent is this? Consider: sudden onset = more urgent, chronic/mild = less urgent, elderly/infant = higher risk.
Step 4 - Specialty Selection: Which medical specialty is best equipped to diagnose and treat this condition?

Example reasoning:
"Patient describes burning urination and lower abdominal pain → urinary system affected → likely UTI or kidney issue → moderate urgency since no fever mentioned → nephrology/urology"

After your reasoning, you MUST end with exactly two lines:
SPECIALTY: <specialty>
SEVERITY: <severity>"""


# Default strategy (change to "one_shot" or "chain_of_thought" as needed)
CLASSIFY_STRATEGY = "zero_shot"


def _clean_response(text: str) -> str:
    """Strip LLM meta-commentary. Keep only the direct message."""
    lines = text.strip().splitlines()

    # Find where actual reply starts
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("hi ") or s.startswith("hello ") or s.startswith("dear ") or s.startswith("namaste"):
            start_idx = i
            break

    # Find where signatures/commentary start at end
    cut_patterns = [
        r"(?i)^(sincerely|best regards|regards|warm regards|yours|thank you,)",
        r"(?i)^\[your name",
        r"(?i)^(--+)$",
        r"(?i)^\*\*key changes",
        r"(?i)^\*\s+\*\*",
        r"(?i)^(note:|p\.?s\.?:)",
        r"(?i)^(the .* team|customer support|support team|hospital team)$",
    ]
    end_idx = len(lines)
    for i in range(len(lines) - 1, start_idx, -1):
        stripped = lines[i].strip()
        if any(re.match(p, stripped) for p in cut_patterns):
            end_idx = i
        elif stripped == "":
            continue
        else:
            break

    skip_patterns = [
        r"(?i)^(okay|sure|here'?s|of course|certainly|absolutely|great|right)[,:]?\s",
        r"(?i)^subject:",
        r"(?i)^re:",
    ]
    cleaned = []
    for line in lines[start_idx:end_idx]:
        stripped = line.strip()
        if any(re.match(p, stripped) for p in skip_patterns):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    return result if result else text.strip()


# ---------------------------------------------------------------------------
# 1. ANALYZE SYMPTOMS — NER + NLTK noun extraction + keyword matching
# ---------------------------------------------------------------------------

def analyze_symptoms(state: AgentState) -> dict:
    """Use the NER-enhanced symptom analyzer to detect specialty from description."""
    result = _symptom_analyzer.analyze(state["description"])
    return {
        "detected_specialty": result["specialty"],
        "severity_hint": result["severity_hint"],
        "symptom_confidence": result["confidence"],
        "matched_keywords": result["matched_keywords"],
        "current_node": "analyze_symptoms",
        "processing_log": [
            f"NER+NLTK symptom analysis → specialty={result['specialty']}, "
            f"severity={result['severity_hint']}, confidence={result['confidence']}, "
            f"keywords={result['matched_keywords']}, "
            f"extracted_terms={result.get('extracted_terms', [])[:10]}"
        ],
    }


# ---------------------------------------------------------------------------
# 2. LLM CLASSIFY — LLM refines specialty and severity
# ---------------------------------------------------------------------------

def llm_classify(state: AgentState) -> dict:
    """Use LLM to classify the medical specialty and severity."""
    specialties_str = ", ".join(SPECIALTIES)
    severities_str = ", ".join(SEVERITY_LEVELS)

    base_prompt = f"""You are a medical triage assistant at {HOSPITAL_NAME}.

PATIENT AGE: {state['patient_age']} years
SYMPTOMS DESCRIBED: {state['description']}
KEYWORD MATCH SUGGESTED: {state['detected_specialty']}

Choose the most appropriate medical specialty from: {specialties_str}
Choose a severity level from: {severities_str}

Consider:
- Chest pain in elderly = high/critical cardiology
- Breathing difficulty = high pulmonology
- Children's issues = pediatrics
- Mental health keywords = psychiatry
- Accidents/unconscious = critical emergency_medicine

Reply with ONLY two lines:
SPECIALTY: <specialty>
SEVERITY: <severity>"""

    # Apply the configured prompt strategy
    if CLASSIFY_STRATEGY == "one_shot":
        prompt = PromptStrategy.one_shot(base_prompt)
    elif CLASSIFY_STRATEGY == "chain_of_thought":
        prompt = PromptStrategy.chain_of_thought(base_prompt)
    else:
        prompt = PromptStrategy.zero_shot(base_prompt)

    raw = _ask_llm(prompt)

    specialty = state["detected_specialty"]  # fallback
    severity = state["severity_hint"]

    for line in raw.splitlines():
        upper = line.strip().upper()
        if upper.startswith("SPECIALTY:"):
            val = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            if val in SPECIALTIES:
                specialty = val
        elif upper.startswith("SEVERITY:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in SEVERITY_LEVELS:
                severity = val

    return {
        "llm_specialty": specialty,
        "llm_severity": severity,
        "current_node": "llm_classify",
        "processing_log": [
            f"LLM classification ({CLASSIFY_STRATEGY}) → specialty={specialty}, severity={severity}",
            f"LLM raw: {raw[:200]}",
        ],
    }


# ---------------------------------------------------------------------------
# 3. ASSESS PATIENT — sentiment, age risk, urgency scoring
# ---------------------------------------------------------------------------

def assess_patient_node(state: AgentState) -> dict:
    """Run patient assessment for urgency scoring."""
    result = assess_patient(state["description"], state["patient_age"])

    urgency = result["urgency_score"]

    # Use LLM severity as a floor — if LLM understood it's critical/high,
    # urgency should not be lower than these thresholds
    llm_severity = state.get("llm_severity", "low")
    severity_floor = {"critical": 8, "high": 6, "moderate": 4, "low": 1}
    floor = severity_floor.get(llm_severity, 1)
    if urgency < floor:
        urgency = floor

    return {
        "sentiment_label": result["sentiment_label"],
        "sentiment_score": result["sentiment_score"],
        "urgency_score": urgency,
        "age_risk_factor": result["age_risk_factor"],
        "current_node": "assess_patient",
        "processing_log": [
            f"Patient assessment → urgency={urgency}/10 (keyword={result['urgency_score']}, "
            f"llm_floor={floor} from severity='{llm_severity}'), "
            f"sentiment={result['sentiment_label']} (score={result['sentiment_score']}), "
            f"age_risk={result['age_risk_factor']}, signals={result['matched_signals']}"
        ],
    }


# ---------------------------------------------------------------------------
# 4. RESOLVE SPECIALTY — pick final specialty from tool + LLM
# ---------------------------------------------------------------------------

def resolve_specialty(state: AgentState) -> dict:
    """Combine tool-based and LLM-based classification to pick final specialty + severity."""
    tool_spec = state.get("detected_specialty", "general_medicine")
    llm_spec = state.get("llm_specialty", tool_spec)  # may not exist if LLM was skipped
    tool_sev = state.get("severity_hint", "low")
    llm_sev = state.get("llm_severity", tool_sev)
    confidence = state.get("symptom_confidence", 0.0)
    urgency = state.get("urgency_score", 3)

    # If deterministic match has high confidence, trust it; LLM is backup
    if confidence >= 0.5 and tool_spec != "general_medicine":
        final_spec = tool_spec
    elif llm_spec and llm_spec != "general_medicine":
        final_spec = llm_spec
    else:
        final_spec = tool_spec

    # Severity: take the higher of (tool hint, LLM, urgency-derived)
    sev_order = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
    urgency_sev = "low"
    if urgency >= 8:
        urgency_sev = "critical"
    elif urgency >= 6:
        urgency_sev = "high"
    elif urgency >= 4:
        urgency_sev = "moderate"

    final_sev = max([tool_sev, llm_sev, urgency_sev], key=lambda s: sev_order.get(s, 1))

    # Override: critical always goes to emergency if not already specialist
    if final_sev == "critical" and final_spec == "general_medicine":
        final_spec = "emergency_medicine"

    return {
        "final_specialty": final_spec,
        "final_severity": final_sev,
        "current_node": "resolve_specialty",
        "processing_log": [
            f"Final resolution → specialty={final_spec} (tool={tool_spec}, llm={llm_spec}), "
            f"severity={final_sev} (tool={tool_sev}, llm={llm_sev}, urgency_derived={urgency_sev})"
        ],
    }


# ---------------------------------------------------------------------------
# 5. FIND DOCTOR — cosine similarity first, then specialty fallback
# ---------------------------------------------------------------------------

def find_doctor(state: AgentState) -> dict:
    """Find the best doctor using cosine similarity on treats, fallback to specialty match."""
    specialty = state["final_specialty"]
    date_str = state["appointment_date"]
    description = state["description"]

    # Get day name from date
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_name = dt.strftime("%A")
    except ValueError:
        day_name = None

    # --- PRIMARY: Cosine similarity on extracted terms vs doctor treats ---
    extracted_terms = _symptom_analyzer.analyze(description).get("extracted_terms", [])
    similarity_results = _doctor_registry.find_by_similarity(extracted_terms, day_name)

    # Try cosine matches first
    for doc, score in similarity_results:
        slots = _appt_manager.get_available_slots(doc.id, date_str, doc.max_patients_per_day)
        if slots:
            return {
                "assigned_doctor_id": doc.id,
                "assigned_doctor_name": doc.name,
                "assigned_doctor_qualification": doc.qualification,
                "assigned_doctor_room": doc.room,
                "final_specialty": doc.specialty,
                "current_node": "find_doctor",
                "processing_log": [
                    f"COSINE MATCH → {doc.name} ({doc.specialty}), "
                    f"similarity={score}, {doc.qualification}, {doc.room}, {len(slots)} slots"
                ],
            }

    # --- FALLBACK: Match by specialty name (always respect day availability) ---
    doctors = _doctor_registry.find_by_specialty(specialty, day_name)

    if not doctors:
        # Try general medicine on the same day
        doctors = _doctor_registry.find_by_specialty("general_medicine", day_name)

    if not doctors:
        return {
            "appointment_status": "no_doctor",
            "current_node": "find_doctor",
            "processing_log": [f"No doctor found for {specialty} on {date_str} ({day_name}) — no one available this day"],
        }

    # Pick doctor with available slots
    for doc in doctors:
        slots = _appt_manager.get_available_slots(doc.id, date_str, doc.max_patients_per_day)
        if slots:
            return {
                "assigned_doctor_id": doc.id,
                "assigned_doctor_name": doc.name,
                "assigned_doctor_qualification": doc.qualification,
                "assigned_doctor_room": doc.room,
                "current_node": "find_doctor",
                "processing_log": [
                    f"SPECIALTY FALLBACK → {doc.name} ({doc.specialty}), "
                    f"{doc.qualification}, {doc.room}, {len(slots)} slots available"
                ],
            }

    return {
        "appointment_status": "no_slot",
        "current_node": "find_doctor",
        "processing_log": [f"All {specialty} doctors fully booked on {date_str}"],
    }


def doctor_found_decision(state: AgentState) -> str:
    """Conditional edge: check if doctor was found."""
    status = state.get("appointment_status", "")
    if status in ("no_doctor", "no_slot"):
        return "no_availability"
    return "schedule"


# ---------------------------------------------------------------------------
# 6. SCHEDULE APPOINTMENT — pick slot based on urgency
# ---------------------------------------------------------------------------

def schedule_appointment(state: AgentState) -> dict:
    """Book the appointment slot. Higher urgency → earlier slot."""
    doctor_id = state["assigned_doctor_id"]
    date_str = state["appointment_date"]
    doc = _doctor_registry.get_doctor(doctor_id)
    if not doc:
        return {
            "appointment_status": "no_doctor",
            "current_node": "schedule",
            "processing_log": ["Error: doctor not found during scheduling"],
        }

    slots = _appt_manager.get_available_slots(doctor_id, date_str, doc.max_patients_per_day)
    if not slots:
        return {
            "appointment_status": "no_slot",
            "current_node": "schedule",
            "processing_log": ["No available slots remaining"],
        }

    urgency = state.get("urgency_score", 5)

    # Pick the first available slot within a preferred time window based on urgency.
    # High urgency → morning (9:00-11:00), Medium → mid-day (10:00-2:00), Low → afternoon (11:00+)
    # This naturally assigns sequential slots as earlier ones get booked.
    if urgency >= 7:
        preferred_start = "09:00"
    elif urgency >= 4:
        preferred_start = "10:00"
    else:
        preferred_start = "11:00"

    # Find first slot at or after preferred start time
    chosen_slot = None
    for slot in slots:
        # Extract hour:minute from slot like "11:30 AM IST"
        time_part = slot.split(" IST")[0].strip()
        parts = time_part.split()
        hm, ampm = parts[0], parts[1]
        h, m = map(int, hm.split(":"))
        if ampm == "PM" and h != 12:
            h += 12
        elif ampm == "AM" and h == 12:
            h = 0
        slot_24h = f"{h:02d}:{m:02d}"

        if slot_24h >= preferred_start:
            chosen_slot = slot
            break

    # If no slot in preferred window, take the first available
    if not chosen_slot:
        chosen_slot = slots[0]

    appt = _appt_manager.book_appointment(
        patient_name=state["patient_name"],
        patient_age=state["patient_age"],
        doctor_id=doctor_id,
        doctor_name=state["assigned_doctor_name"],
        specialty=state["final_specialty"],
        date=date_str,
        time_slot=chosen_slot,
        room=doc.room,
        urgency_score=urgency,
        severity=state.get("final_severity", ""),
        sentiment_label=state.get("sentiment_label", ""),
        sentiment_score=state.get("sentiment_score", 0.0),
    )

    return {
        "appointment_id": appt.appointment_id,
        "time_slot": chosen_slot,
        "appointment_status": "confirmed",
        "current_node": "schedule",
        "processing_log": [
            f"Appointment BOOKED → {appt.appointment_id}, "
            f"slot={chosen_slot}, urgency={urgency}/10"
        ],
    }


# ---------------------------------------------------------------------------
# 7. GENERATE CONFIRMATION — LLM writes the patient message
# ---------------------------------------------------------------------------

def generate_confirmation(state: AgentState) -> dict:
    """Generate a confirmation message for the patient."""
    specialty_display = state["final_specialty"].replace("_", " ").title()

    prompt = f"""You are the appointment assistant at {HOSPITAL_NAME}.
Write a short confirmation message for the patient.

PATIENT NAME: {state['patient_name']}
AGE: {state['patient_age']} years
SYMPTOMS: {state['description'][:200]}
SEVERITY: {state['final_severity']}

APPOINTMENT DETAILS:
- Appointment ID: {state['appointment_id']}
- Doctor: {state['assigned_doctor_name']}
- Qualification: {state.get('assigned_doctor_qualification', '')}
- Specialty: {specialty_display}
- Date: {state['appointment_date']}
- Time: {state['time_slot']}
- Location: {state.get('assigned_doctor_room', '')}, {HOSPITAL_NAME}

Output ONLY the confirmation message. Start with "Hi {state['patient_name']}," directly.
Include all appointment details clearly. Add one line about what to bring (ID proof, past reports).
If severity is high/critical, mention to come early or visit emergency if condition worsens.
Keep under 120 words. No subject lines, no signatures, no commentary.

Hi {state['patient_name']},"""

    raw = _ask_llm(prompt)
    message = _clean_response(raw)

    # Ensure it starts with greeting
    if not message.lower().startswith("hi "):
        message = f"Hi {state['patient_name']},\n\n{message}"

    return {
        "response_message": message,
        "current_node": "confirm",
        "processing_log": ["Confirmation message generated"],
    }


# ---------------------------------------------------------------------------
# 8. NO AVAILABILITY — handle no doctor/slot case
# ---------------------------------------------------------------------------

def handle_no_availability(state: AgentState) -> dict:
    """Generate a message when no doctor or slot is available."""
    status = state.get("appointment_status", "no_slot")
    specialty_display = state.get("final_specialty", "general medicine").replace("_", " ").title()

    if status == "no_doctor":
        msg = (
            f"Hi {state['patient_name']},\n\n"
            f"We're sorry, but we currently don't have a {specialty_display} specialist "
            f"available on {state['appointment_date']} at {HOSPITAL_NAME}.\n\n"
            f"Please try a different date, or call us at {HOSPITAL_PHONE} "
            f"(Mon-Sat, 9 AM - 5 PM IST) and we'll help you find the earliest available slot.\n\n"
            f"If your condition is urgent, please visit our Emergency department directly."
        )
    else:
        msg = (
            f"Hi {state['patient_name']},\n\n"
            f"All {specialty_display} appointment slots are fully booked for "
            f"{state['appointment_date']} at {HOSPITAL_NAME}.\n\n"
            f"Please try the next available date, or call us at {HOSPITAL_PHONE} "
            f"(Mon-Sat, 9 AM - 5 PM IST) for waitlist options.\n\n"
            f"If your condition is urgent, please visit our Emergency department directly."
        )

    return {
        "response_message": msg,
        "current_node": "no_availability",
        "processing_log": [f"No availability — {status} for {specialty_display}"],
    }
