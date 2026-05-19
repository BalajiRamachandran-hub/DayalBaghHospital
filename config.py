"""Configuration for the DayalBagh Medical Appointment Agent."""
import os

# Ollama model configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Agent constraints
MAX_RETRIES = 2
HOSPITAL_NAME = "DayalBagh Group Of Hospitals"
HOSPITAL_LOCATION = "DayalBagh, Agra, Uttar Pradesh"
HOSPITAL_PHONE = "7904025851" # Manager's contact number

# PostgreSQL configuration
DB_NAME = os.getenv("DB_NAME", "dayalbagh_hospital")
DB_USER = os.getenv("DB_USER", "b0r06cj")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# Appointment hours (IST)
WORKING_HOURS_START = 9   # 9:00 AM IST
WORKING_HOURS_END = 17    # 5:00 PM IST
SLOT_DURATION_MINUTES = 30

# Severity levels (higher = more urgent, gets earlier slot)
SEVERITY_LEVELS = ["low", "moderate", "high", "critical"]

# Supported specialties
SPECIALTIES = [
    "general_medicine",
    "cardiology",
    "neurology",
    "orthopedics",
    "gastroenterology",
    "pulmonology",
    "dermatology",
    "ent",
    "ophthalmology",
    "gynecology",
    "pediatrics",
    "psychiatry",
    "urology",
    "oncology",
    "emergency_medicine",
]
