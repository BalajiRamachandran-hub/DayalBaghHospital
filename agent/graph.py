"""LangGraph workflow definition for the Medical Appointment Agent."""

from langgraph.graph import END, StateGraph

from agent.state import AgentState
from agent.nodes import (
    analyze_symptoms,
    llm_classify,
    assess_patient_node,
    resolve_specialty,
    find_doctor,
    doctor_found_decision,
    schedule_appointment,
    generate_confirmation,
    handle_no_availability,
)


def build_graph() -> StateGraph:
    """Construct and compile the appointment agent workflow.

    Workflow:
        analyze_symptoms > llm_classify > assess_patient > resolve_specialty
        > find_doctor ----> schedule > confirm > END
                      ----> no_availability > END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze_symptoms", analyze_symptoms)
    graph.add_node("llm_classify", llm_classify)
    graph.add_node("assess_patient", assess_patient_node)
    graph.add_node("resolve_specialty", resolve_specialty)
    graph.add_node("find_doctor", find_doctor)
    graph.add_node("schedule", schedule_appointment)
    graph.add_node("confirm", generate_confirmation)
    graph.add_node("no_availability", handle_no_availability)

    # Linear edges
    graph.set_entry_point("analyze_symptoms")
    graph.add_edge("analyze_symptoms", "llm_classify")
    graph.add_edge("llm_classify", "assess_patient")
    graph.add_edge("assess_patient", "resolve_specialty")
    graph.add_edge("resolve_specialty", "find_doctor")

    # Conditional: doctor found → schedule or no_availability
    graph.add_conditional_edges(
        "find_doctor",
        doctor_found_decision,
        {
            "schedule": "schedule",
            "no_availability": "no_availability",
        },
    )

    graph.add_edge("schedule", "confirm")

    # Terminal
    graph.add_edge("confirm", END)
    graph.add_edge("no_availability", END)

    return graph.compile()


# Singleton
appointment_agent = build_graph()
