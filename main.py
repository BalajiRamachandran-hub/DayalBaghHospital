#!/usr/bin/env python3
"""Main entry point — interactive medical appointment chatbot + batch evaluation."""

import sys
import time
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from agent.graph import appointment_agent
from agent.nodes import get_appointment_manager
from evaluation.evaluator import AgentEvaluation, evaluate_result
from config import HOSPITAL_NAME

console = Console()


def _default_date() -> str:
    """Return tomorrow's date as YYYY-MM-DD."""
    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


def _build_initial_state(name: str, age: int, description: str, date: str) -> dict:
    return {
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


def run_agent(name: str, age: int, description: str, date: str) -> tuple[dict, float]:
    """Run the appointment agent and return (result, elapsed)."""
    state = _build_initial_state(name, age, description, date)
    start = time.time()
    try:
        result = appointment_agent.invoke(state)
    except Exception as exc:
        elapsed = time.time() - start
        state["error"] = str(exc)
        state["appointment_status"] = "error"
        return state, elapsed
    return result, time.time() - start


def display_result(result: dict, elapsed: float) -> None:
    """Pretty-print one appointment result."""
    status = result.get("appointment_status", "unknown")
    color = {"confirmed": "green", "no_slot": "yellow", "no_doctor": "yellow", "error": "red"}.get(status, "white")

    specialty_display = result.get("final_specialty", "?").replace("_", " ").title()
    severity = result.get("final_severity", "?")
    sev_color = {"critical": "red", "high": "yellow", "moderate": "cyan", "low": "green"}.get(severity, "white")

    header = Text()
    header.append(f"Patient: {result.get('patient_name', '?')}", style="bold cyan")
    header.append(f"  |  Age: {result.get('patient_age', '?')}")
    header.append(f"  |  Status: ")
    header.append(status.upper(), style=f"bold {color}")
    header.append(f"  |  Time: {elapsed:.1f}s")

    console.print()
    console.print(Panel(header, box=box.HEAVY))

    # Classification info
    console.print(f"  [bold]Specialty:[/bold]  {specialty_display}")
    console.print(f"  [bold]Severity:[/bold]   [{sev_color}]{severity}[/{sev_color}]")
    console.print(f"  [bold]Urgency:[/bold]    {result.get('urgency_score', '?')}/10")
    console.print(f"  [bold]Sentiment:[/bold]  {result.get('sentiment_label', '?')} (score={result.get('sentiment_score', 0):.2f})")
    console.print(f"  [bold]Keywords:[/bold]   {', '.join(result.get('matched_keywords', []))}")

    if status == "confirmed":
        console.print(f"  [bold]Doctor:[/bold]     {result.get('assigned_doctor_name', '?')}")
        console.print(f"  [bold]Room:[/bold]       {result.get('assigned_doctor_room', '?')}")
        console.print(f"  [bold]Slot:[/bold]       {result.get('time_slot', '?')}")
        console.print(f"  [bold]Appt ID:[/bold]   {result.get('appointment_id', '?')}")

    # Response message
    msg = result.get("response_message", "")
    if msg:
        console.print()
        console.print(Panel(msg, title="Agent Response", border_style=color, padding=(1, 2)))

    # Processing log
    log = result.get("processing_log", [])
    if log:
        console.print()
        console.print("  [dim]Processing Log:[/dim]")
        for entry in log:
            console.print(f"    [dim]• {entry}[/dim]")


def display_evaluation(evaluation: AgentEvaluation) -> None:
    """Display aggregate evaluation."""
    console.print()
    console.print(Panel("[bold]AGENT EVALUATION REPORT[/bold]", box=box.DOUBLE, style="magenta"))

    summary = evaluation.summary_dict()
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Patients", str(summary["total_patients"]))
    table.add_row("Booking Success Rate", summary["booking_rate"])
    table.add_row("Error Rate", summary["error_rate"])
    table.add_row("Avg Urgency Score", summary["avg_urgency"])
    table.add_row("Avg Processing Time", summary["avg_processing_time"])
    console.print(table)

    # Per-patient table
    detail = Table(title="Per-Patient Breakdown", box=box.ROUNDED)
    detail.add_column("Patient", style="cyan")
    detail.add_column("Age", justify="center")
    detail.add_column("Specialty")
    detail.add_column("Severity")
    detail.add_column("Urgency", justify="center")
    detail.add_column("Doctor")
    detail.add_column("Slot")
    detail.add_column("Status", justify="center")
    detail.add_column("Time", justify="right")

    for r in evaluation.results:
        s_color = {"confirmed": "green", "no_slot": "yellow", "no_doctor": "yellow"}.get(r.status, "red")
        detail.add_row(
            r.patient_name, str(r.patient_age),
            r.final_specialty.replace("_", " ").title(), r.severity,
            str(r.urgency_score), r.doctor_assigned,
            r.time_slot, Text(r.status, style=s_color),
            f"{r.processing_time:.1f}s",
        )
    console.print(detail)

    # Schedule summary
    appt_mgr = get_appointment_manager()
    if evaluation.results:
        date = evaluation.results[0].time_slot  # just show the schedule
    console.print()


def display_schedule(date: str) -> None:
    """Show the day's appointment schedule."""
    appt_mgr = get_appointment_manager()
    appts = appt_mgr.get_appointments_for_date(date)
    if not appts:
        return

    console.print()
    sched = Table(title=f"Appointment Schedule — {date}", box=box.ROUNDED)
    sched.add_column("Appt ID", style="cyan")
    sched.add_column("Time")
    sched.add_column("Patient")
    sched.add_column("Age", justify="center")
    sched.add_column("Doctor")
    sched.add_column("Specialty")
    sched.add_column("Room")
    sched.add_column("Urgency", justify="center")

    appts_sorted = sorted(appts, key=lambda a: a.time_slot)
    for a in appts_sorted:
        sched.add_row(
            a.appointment_id, a.time_slot, a.patient_name,
            str(a.patient_age), a.doctor_name,
            a.specialty.replace("_", " ").title(),
            a.room, str(a.urgency_score),
        )
    console.print(sched)


# ── Interactive Mode ──

def interactive_mode() -> None:
    console.print(
        Panel(
            f"[bold cyan]{HOSPITAL_NAME}[/bold cyan]\n"
            f"Medical Appointment Assistant\n"
            f"Powered by LangGraph + Ollama (Gemma 3)\n\n"
            f"[dim]Describe your symptoms and I'll book an appointment with the right specialist.\n"
            f"Type 'schedule' to view today's appointments. Type 'quit' to exit.[/dim]",
            box=box.DOUBLE_EDGE, padding=(1, 4),
        )
    )

    # Get appointment date
    default_date = _default_date()
    console.print(f"\n[bold]Appointment date[/bold] [dim](press Enter for {default_date}):[/dim] ", end="")
    date_input = input().strip()
    appt_date = date_input if date_input else default_date
    console.print(f"Booking appointments for: [bold]{appt_date}[/bold]\n")

    while True:
        console.print("[bold cyan]─[/bold cyan]" * 60)
        try:
            name = console.input("\n[bold]Patient Name:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not name:
            continue
        if name.lower() in ("quit", "exit", "q", "bye"):
            console.print(f"\n[bold cyan]Thank you for using {HOSPITAL_NAME} appointment system![/bold cyan]\n")
            display_schedule(appt_date)
            break
        if name.lower() == "schedule":
            display_schedule(appt_date)
            continue

        try:
            age_str = console.input("[bold]Age:[/bold] ").strip()
            age = int(age_str)
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("[red]Please enter a valid age.[/red]")
            continue

        try:
            description = console.input("[bold green]Describe your symptoms:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not description:
            console.print("[red]Please describe your symptoms.[/red]")
            continue

        console.print(f"\n[dim]Analyzing symptoms and booking appointment...[/dim]\n")

        result, elapsed = run_agent(name, age, description, appt_date)
        display_result(result, elapsed)
        console.print()


# ── Batch Mode ──

SAMPLE_PATIENTS = [
    ("Ramesh Agarwal", 65, "I am having severe chest pain since morning and difficulty breathing. It feels like a heavy weight on my chest. I am very scared."),
    ("Priya Sharma", 28, "I have been having terrible headaches for the past week. Sometimes I feel dizzy and my vision gets blurry. Getting worse."),
    ("Amit Kumar", 35, "I had an accident on my bike and my knee is swollen badly. I can't walk properly. There might be a fracture."),
    ("Lakshmi Devi", 72, "I have been coughing continuously for 3 weeks. There is blood in my cough sometimes. I am very weak and have lost weight."),
    ("Suresh Yadav", 45, "Having stomach pain and acidity problems. Vomiting after every meal. This has been going on for months."),
    ("Ananya Gupta", 8, "My daughter has had high fever for 3 days. She has a bad cough and cold. She can't eat anything."),
    ("Vikas Tiwari", 40, "I have been feeling very depressed and can't sleep at all. Having anxiety attacks. Feeling suicidal sometimes."),
    ("Neha Singh", 30, "I have itching and rashes all over my body. It started a week ago and is spreading. Some pimples are also there."),
    ("Radhika Verma", 55, "I have burning sensation while urinating and there is blood in urine. Severe pain in lower abdomen and back. Might be kidney stone."),
    ("Mohit Rastogi", 22, "I just want a routine health checkup. Nothing specific, just general wellness check."),
]


def batch_mode() -> None:
    console.print(
        Panel(
            f"[bold cyan]{HOSPITAL_NAME}[/bold cyan]\n"
            f"Batch Appointment Processing + Evaluation\n"
            f"Powered by LangGraph + Ollama (Gemma 3)",
            box=box.DOUBLE_EDGE, padding=(1, 4),
        )
    )

    appt_date = _default_date()
    console.print(f"\nProcessing [bold]{len(SAMPLE_PATIENTS)}[/bold] patients for [bold]{appt_date}[/bold]\n")

    evaluation = AgentEvaluation()

    for name, age, desc in SAMPLE_PATIENTS:
        console.print(f"[bold]Processing: {name} (age {age})[/bold] ...")
        result, elapsed = run_agent(name, age, desc, appt_date)
        display_result(result, elapsed)
        eval_result = evaluate_result(result, elapsed)
        evaluation.results.append(eval_result)

    display_evaluation(evaluation)
    display_schedule(appt_date)

    # Improvements
    console.print()
    console.print(
        Panel(
            "[bold]Suggested Improvements[/bold]\n\n"
            "1. [bold]Larger LLM:[/bold] Use Llama 3 8B for more accurate symptom classification\n"
            "2. [bold]Medical NER:[/bold] Add named entity recognition for precise symptom extraction\n"
            "3. [bold]EHR Integration:[/bold] Connect to Electronic Health Records for patient history\n"
            "4. [bold]Multi-turn Chat:[/bold] Support follow-up questions for better symptom clarity\n"
            "5. [bold]SMS/WhatsApp:[/bold] Send appointment confirmations via SMS or WhatsApp\n"
            "6. [bold]Waitlist:[/bold] Auto-manage waitlists when slots are full\n"
            "7. [bold]Feedback Loop:[/bold] Collect post-visit feedback to improve triage accuracy",
            border_style="blue", padding=(1, 2),
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_mode()
    else:
        interactive_mode()
