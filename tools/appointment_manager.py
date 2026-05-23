"""Appointment manager — handles slot booking for a given date."""

from dataclasses import dataclass, field
from datetime import datetime

import psycopg2

from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT


@dataclass
class Appointment:
    appointment_id: str
    patient_name: str
    patient_age: int
    doctor_id: str
    doctor_name: str
    specialty: str
    date: str          # YYYY-MM-DD
    time_slot: str     # e.g., "09:00 AM IST"
    room: str
    urgency_score: int
    status: str = "confirmed"
    severity: str = ""
    sentiment_label: str = ""
    sentiment_score: float = 0.0


def _get_db_connection():
    """Create a new database connection."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


class AppointmentManager:
    """Manages appointment slots for a given date."""

    def __init__(self):
        # doctor_id -> date -> list of booked time slots
        self._bookings: dict[str, dict[str, list[str]]] = {}
        self._appointments: list[Appointment] = []
        self._counter = self._get_max_counter()
        self._load_existing_bookings()

    def _get_max_counter(self) -> int:
        """Get the highest appointment counter from the database."""
        try:
            conn = _get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT MAX(CAST(SUBSTRING(appointment_id FROM 5) AS INTEGER)) FROM appointments")
            result = cur.fetchone()[0]
            cur.close()
            conn.close()
            return result if result else 0
        except Exception:
            return 0

    def _load_existing_bookings(self) -> None:
        """Load already-booked slots from the database so we don't double-book after restart."""
        try:
            conn = _get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT doctor_id, appointment_date, time_slot FROM appointments WHERE status = 'confirmed'"
            )
            for doctor_id, appt_date, time_slot in cur.fetchall():
                date_str = appt_date if isinstance(appt_date, str) else appt_date.strftime("%Y-%m-%d")
                self._bookings.setdefault(doctor_id, {}).setdefault(date_str, []).append(time_slot)
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[DB WARNING] Could not load existing bookings: {e}")

    def _get_all_slots(self) -> list[str]:
        """Generate 30-minute slots from 9:00 AM to 5:00 PM IST."""
        slots = []
        for hour in range(9, 17):
            for minute in (0, 30):
                h12 = hour if hour <= 12 else hour - 12
                ampm = "AM" if hour < 12 else "PM"
                slots.append(f"{h12:02d}:{minute:02d} {ampm} IST")
        return slots

    def get_available_slots(self, doctor_id: str, date: str, max_patients: int) -> list[str]:
        """Return available time slots for a doctor on a given date."""
        all_slots = self._get_all_slots()
        booked = self._bookings.get(doctor_id, {}).get(date, [])

        if len(booked) >= max_patients:
            return []

        return [s for s in all_slots if s not in booked]

    def _save_to_db(self, appt: Appointment) -> None:
        """Persist appointment to PostgreSQL."""
        try:
            conn = _get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO appointments
                   (appointment_id, patient_name, patient_age, doctor_id, doctor_name,
                    specialty, severity, appointment_date, time_slot, room,
                    urgency_score, sentiment_label, sentiment_score, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    appt.appointment_id,
                    appt.patient_name,
                    appt.patient_age,
                    appt.doctor_id,
                    appt.doctor_name,
                    appt.specialty,
                    appt.severity,
                    appt.date,
                    appt.time_slot,
                    appt.room,
                    appt.urgency_score,
                    appt.sentiment_label,
                    appt.sentiment_score,
                    appt.status,
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"[DB WARNING] Could not save to PostgreSQL: {e}")

    def book_appointment(
        self, patient_name: str, patient_age: int,
        doctor_id: str, doctor_name: str, specialty: str,
        date: str, time_slot: str, room: str, urgency_score: int,
        severity: str = "", sentiment_label: str = "", sentiment_score: float = 0.0,
    ) -> Appointment:
        """Book an appointment and return the confirmation."""
        self._counter += 1
        appt_id = f"APT-{self._counter:04d}"

        # Record booking
        self._bookings.setdefault(doctor_id, {}).setdefault(date, []).append(time_slot)

        appt = Appointment(
            appointment_id=appt_id,
            patient_name=patient_name,
            patient_age=patient_age,
            doctor_id=doctor_id,
            doctor_name=doctor_name,
            specialty=specialty,
            date=date,
            time_slot=time_slot,
            room=room,
            urgency_score=urgency_score,
            severity=severity,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
        )
        self._appointments.append(appt)
        self._save_to_db(appt)
        return appt

    def get_appointments_for_date(self, date: str) -> list[Appointment]:
        """Get all appointments for a given date."""
        return [a for a in self._appointments if a.date == date]

    def get_schedule_summary(self, date: str) -> dict:
        """Summary of appointments on a date."""
        appts = self.get_appointments_for_date(date)
        by_specialty: dict[str, int] = {}
        by_doctor: dict[str, int] = {}
        for a in appts:
            by_specialty[a.specialty] = by_specialty.get(a.specialty, 0) + 1
            by_doctor[a.doctor_name] = by_doctor.get(a.doctor_name, 0) + 1
        return {
            "date": date,
            "total_appointments": len(appts),
            "by_specialty": by_specialty,
            "by_doctor": by_doctor,
        }
