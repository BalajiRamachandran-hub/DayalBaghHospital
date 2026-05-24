# DayalBagh Group Of Hospitals — AI Appointment Booking System

An autonomous AI agent built with **LangGraph**, **NLTK**, **scikit-learn (Cosine Similarity)**, **Ollama (Llama 3.1 8B)**, **PostgreSQL**, **Tableau**, and **Streamlit** that triages patients using NER-based symptom extraction, matches doctors via TF-IDF cosine similarity, assesses urgency, and books specialist appointments with priority-based scheduling.

---

## Features

- **NER + NLTK Pipeline** — Extracts medical nouns, adjectives, bigrams, and trigrams from patient descriptions
- **Cosine Similarity Matching** — TF-IDF vectorization of doctor `treats[]` lists; matches patients to best-fit doctor (threshold ≥ 0.15)
- **Conditional LLM Skip** — When NER confidence ≥ 0.5, the LLM is bypassed entirely (< 1s vs ~8s)
- **LLM Fallback** — Llama 3.1 8B via Ollama used only when deterministic matching is uncertain
- **Urgency Scoring** — Hybrid approach combining keyword sentiment, age risk, and LLM severity (1-10 scale)
- **Priority Scheduling** — Critical patients get the earliest available slots; sequential within time windows
- **15 Medical Specialties** — Cardiology, Neurology, Orthopedics, Oncology, Emergency Medicine, etc.
- **17 Doctors** — Each with qualifications, `treats[]` keywords, room assignments, and daily patient limits
- **PostgreSQL Persistence** — All appointments stored in a database; loads on startup to prevent double-booking
- **Tableau Analytics** — Connected to PostgreSQL for specialty demand, urgency distribution, and workload dashboards
- **Streamlit Web UI** — Chat interface with real-time appointment dashboard
- **Docker Support** — Containerized deployment with volume mounts for development

---

## Architecture

```
Patient Input → [1] Analyze Symptoms (NER + NLTK)
                    ├── confidence ≥ 0.5 → SKIP LLM ──────────────────────┐
                    └── confidence < 0.5 → [2] LLM Classify (Llama 3.1)  │
             → [3] Assess Patient (age + sentiment + urgency)            │
             → [4] Resolve Specialty                                     │
             → [5] Find Doctor (Cosine Similarity + day constraint) ◄────┘
                    ├── doctor found → [6] Schedule Slot → [7] Confirm (LLM)
                    └── no doctor   → [8] Handle No Availability
```

**8-node LangGraph state machine** with 2 conditional edges:
1. **LLM Skip** — If NER confidence ≥ 0.5 and specialty ≠ general_medicine, bypass the LLM
2. **Availability Check** — Routes to scheduling or no-availability handler

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| Ollama | Latest | Local LLM server |
| PostgreSQL | 14+ | Appointment database |
| Docker | Optional | Containerized deployment |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/BalajiRamachandran-hub/DayalBaghHospital.git
cd DayalBaghHospital
```

### 2. Install Ollama and pull the model

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama
ollama serve

# Pull Llama 3.1 8B (in another terminal)
ollama pull llama3.1:8b
```

### 3. Set up PostgreSQL

```bash
# Install PostgreSQL (macOS)
brew install postgresql@17
brew services start postgresql@17

# Create the database
createdb dayalbagh_hospital

# Create the appointments table
psql -d dayalbagh_hospital -c "
CREATE TABLE IF NOT EXISTS appointments (
    id SERIAL PRIMARY KEY,
    appointment_id VARCHAR(20) UNIQUE NOT NULL,
    patient_name VARCHAR(100) NOT NULL,
    patient_age INTEGER,
    doctor_id VARCHAR(20),
    doctor_name VARCHAR(100),
    specialty VARCHAR(50),
    severity VARCHAR(20),
    appointment_date DATE,
    time_slot VARCHAR(30),
    room VARCHAR(20),
    urgency_score INTEGER,
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    status VARCHAR(20) DEFAULT 'confirmed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure (optional)

Edit `config.py` or use environment variables:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export DB_NAME="dayalbagh_hospital"
export DB_USER="your_username"
export DB_PASSWORD=""
export DB_HOST="localhost"
export DB_PORT="5432"
```

### 6. Run the app

**Option A — Streamlit Web UI (recommended):**
```bash
streamlit run streamlit_app.py --server.headless true
```
Open http://localhost:8501

**Option B — Command-line:**
```bash
python main.py
```

---

## Docker Deployment

### Build the image

```bash
docker build -t dayalbagh-hospital .
```

### Run in detached mode

```bash
docker run -d --name hospital-app \
  -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e DB_HOST=host.docker.internal \
  -e DB_USER=your_username \
  dayalbagh-hospital
```

### Development mode (with volume mount — no rebuild needed)

```bash
docker run -d --name hospital-app \
  -p 8501:8501 \
  -v $(pwd):/app \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e DB_HOST=host.docker.internal \
  dayalbagh-hospital
```

After code changes, just:
```bash
docker restart hospital-app
```

### Useful commands

```bash
docker logs hospital-app        # View logs
docker logs -f hospital-app     # Follow logs live
docker stop hospital-app        # Stop
docker rm hospital-app          # Remove container
```

---

## Project Structure

```
├── config.py                  # Central configuration (model, DB, hours, specialties)
├── main.py                    # CLI entry point
├── streamlit_app.py           # Web UI with chat + dashboard
├── generate_pptx.py           # Presentation generator script
├── Dockerfile                 # Container build file
├── requirements.txt           # Python dependencies
├── agent/
│   ├── graph.py               # LangGraph pipeline (8 nodes, 2 conditional edges)
│   ├── nodes.py               # Node functions + PromptStrategy + LLM helpers
│   └── state.py               # AgentState TypedDict (25+ fields)
├── tools/
│   ├── symptom_analyzer.py    # NER + NLTK symptom extraction & keyword matching
│   ├── patient_assessor.py    # Urgency scoring (sentiment + age risk)
│   ├── doctor_registry.py     # TF-IDF + Cosine Similarity doctor matching
│   └── appointment_manager.py # Slot booking + PostgreSQL persistence
├── data/
│   ├── symptom_map.json       # 15 categories, 25+ keywords each
│   └── doctors.json           # 17 doctors with treats[] lists (15-24 keywords each)
└── evaluation/
    └── evaluator.py           # Agent evaluation framework
```

---

## How Urgency Scoring Works

The system uses a **hybrid approach**:

1. **Keyword scoring** — Scans patient description for distress words ("accident", "blood loss", "dying", "can't breathe") and assigns weighted scores
2. **LLM severity floor** — If the LLM classifies the case as critical/high but keywords missed it, urgency is bumped to a minimum threshold:

| LLM Severity | Minimum Urgency |
|-------------|-----------------|
| Critical | 8 |
| High | 6 |
| Moderate | 4 |
| Low | 1 |

3. **Age risk factor** — Infants (<5) and elderly (75+) get additional urgency boost
4. **Slot assignment** — Urgency ≥ 7 gets the earliest slot, 4-6 gets early-mid, below 4 gets mid-day

---

## Tech Stack

- **LangGraph (v0.2+)** — State machine orchestration with conditional edges
- **NLTK** — Tokenization, POS tagging, NER (ne_chunk), lemmatization, n-grams
- **scikit-learn** — TF-IDF Vectorizer + Cosine Similarity for doctor matching
- **Ollama + Llama 3.1 8B** — Local LLM fallback for classification and confirmation
- **PostgreSQL** — Persistent appointment storage with startup loading
- **Tableau Desktop** — Visual analytics connected to PostgreSQL
- **Streamlit** — Web UI framework
- **Docker** — Containerized deployment
- **Python 3.13** — Runtime

---

## License

MIT
