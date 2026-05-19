"""Streamlit UI for the DayalBagh Hospital Medical Appointment Agent."""

import streamlit as st
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

from agent.graph import appointment_agent
from agent.nodes import get_appointment_manager
from config import HOSPITAL_NAME, DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

# --- Page Config ---
st.set_page_config(
    page_title=f"{HOSPITAL_NAME} - Appointment Agent",
    page_icon="🏥",
    layout="wide",
)

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "booking_step" not in st.session_state:
    st.session_state.booking_step = None


def get_db_appointments(date_filter=None):
    """Fetch appointments from PostgreSQL, filtered by date."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT,
        )
        cur = conn.cursor()
        if date_filter:
            cur.execute(
                "SELECT appointment_id, patient_name, patient_age, doctor_name, "
                "specialty, severity, appointment_date, time_slot, room, "
                "urgency_score, sentiment_label, created_at "
                "FROM appointments WHERE appointment_date = %s ORDER BY time_slot ASC",
                (date_filter,),
            )
        else:
            cur.execute(
                "SELECT appointment_id, patient_name, patient_age, doctor_name, "
                "specialty, severity, appointment_date, time_slot, room, "
                "urgency_score, sentiment_label, created_at "
                "FROM appointments ORDER BY created_at DESC"
            )
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame()


def run_agent(name: str, age: int, description: str, date: str) -> dict:
    """Run the appointment agent and return state."""
    state = {
        "patient_name": name,
        "patient_age": age,
        "description": description,
        "appointment_date": date,
        "detected_specialty": "",
        "severity_hint": "",
        "symptom_confidence": 0.0,
        "matched_keywords": [],
        "sentiment_label": "",
        "sentiment_score": 0.0,
        "urgency_score": 0,
        "age_risk_factor": 0,
        "llm_specialty": "",
        "llm_severity": "",
        "final_specialty": "",
        "final_severity": "",
        "assigned_doctor_id": "",
        "assigned_doctor_name": "",
        "assigned_doctor_qualification": "",
        "assigned_doctor_room": "",
        "appointment_id": "",
        "time_slot": "",
        "appointment_status": "",
        "response_message": "",
        "processing_log": [],
        "current_node": "start",
        "error": "",
    }
    try:
        result = appointment_agent.invoke(state)
        return result
    except Exception as exc:
        state["error"] = str(exc)
        state["appointment_status"] = "error"
        return state


# --- Sidebar: Appointment History ---
with st.sidebar:
    st.header("📋 Appointment History")

    selected_date = st.date_input("Filter by date", value=datetime.now().date())
    df = get_db_appointments(date_filter=str(selected_date))

    if not df.empty:
        st.metric(f"Appointments on {selected_date}", len(df))

        col1, col2 = st.columns(2)
        with col1:
            severity_counts = df["severity"].value_counts()
            st.bar_chart(severity_counts, color="#FF6B6B")
        with col2:
            specialty_counts = df["specialty"].value_counts().head(5)
            st.bar_chart(specialty_counts, color="#4ECDC4")

        st.subheader("Recent Bookings")
        st.dataframe(
            df[["patient_name", "doctor_name", "specialty", "time_slot", "severity"]].head(10),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No appointments booked yet. Start by booking one!")

    if st.button("🔄 Refresh Data"):
        st.rerun()


# --- Main Area ---
st.title(f"🏥 {HOSPITAL_NAME}")
st.caption("AI-Powered Medical Appointment Assistant")

# --- Chat Messages Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Booking Form ---
if st.session_state.booking_step is None:
    with st.chat_message("assistant"):
        st.markdown(
            "Welcome! I'll help you book a medical appointment.\n\n"
            "Please fill in the patient details below:"
        )

    with st.form("patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="e.g., Ramesh Agarwal")
            patient_age = st.number_input("Age", min_value=1, max_value=120, value=30)
        with col2:
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            appointment_date = st.date_input(
                "Appointment Date",
                value=datetime.now() + timedelta(days=1),
                min_value=datetime.now(),
            )

        symptoms = st.text_area(
            "Describe your symptoms",
            placeholder="e.g., Severe chest pain since morning, difficulty breathing, feeling scared...",
            height=100,
        )

        submitted = st.form_submit_button("🩺 Book Appointment", use_container_width=True)

        if submitted:
            if not patient_name or not symptoms:
                st.error("Please fill in patient name and symptoms.")
            else:
                # Add user message
                user_msg = f"**Patient:** {patient_name} (Age: {patient_age})\n\n**Symptoms:** {symptoms}\n\n**Date:** {appointment_date}"
                st.session_state.messages.append({"role": "user", "content": user_msg})

                # Run the agent
                with st.spinner("🔍 Analyzing symptoms and booking appointment..."):
                    result = run_agent(
                        name=patient_name,
                        age=patient_age,
                        description=symptoms,
                        date=str(appointment_date),
                    )

                # Build response
                status = result.get("appointment_status", "error")

                if status == "confirmed":
                    response = f"""✅ **Appointment Confirmed!**

| Detail | Info |
|--------|------|
| 🆔 Appointment ID | `{result.get('appointment_id')}` |
| 👨‍⚕️ Doctor | {result.get('assigned_doctor_name')} |
| 🏥 Specialty | {result.get('final_specialty', '').replace('_', ' ').title()} |
| 📅 Date | {result.get('appointment_date')} |
| ⏰ Time | {result.get('time_slot')} |
| 📍 Room | {result.get('assigned_doctor_room')} |
| ⚠️ Severity | {result.get('final_severity', '').title()} |
| 🔴 Urgency | {result.get('urgency_score')}/10 |
| 💭 Sentiment | {result.get('sentiment_label', '').title()} |

---

{result.get('response_message', '')}"""
                elif status == "error":
                    response = f"❌ **Error:** {result.get('error', 'Unknown error occurred')}"
                else:
                    response = result.get("response_message", "No availability for the requested date.")

                st.session_state.messages.append({"role": "assistant", "content": response})

                # Show processing log in expander
                if result.get("processing_log"):
                    with st.expander("🔬 Processing Steps (click to expand)"):
                        for step in result["processing_log"]:
                            st.code(step, language=None)

                st.rerun()

# --- Footer ---
st.divider()
st.caption(
    f"Powered by LangGraph + Llama 3.1 8B (Ollama) | "
    f"Data stored in PostgreSQL | "
    f"Built for {HOSPITAL_NAME}"
)
