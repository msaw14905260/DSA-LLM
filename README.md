# DSA-LLM: Homeless Services Recommendation System

A RAG (Retrieval-Augmented Generation) system that helps case managers find relevant homeless services in San Diego County. Combines semantic search, multi-label classification, and LLM-powered responses to match clients with the right services.

Built on a dataset of **1,719 services** from the 211 San Diego database, spanning 22 service types across housing, food, mental health, and more.

## How It Works

```
User Query
    |
    v
[Classify Intent] --> Detect which of 22 service types the query is about
    |
    v
[Retrieve from ChromaDB] --> Semantic search + type-match re-ranking
    |
    v
[Generate Response] --> Llama 3.2 produces case manager recommendations
```

**Three retrieval strategies:**
- **Basic** — pure semantic search (cosine similarity on embeddings)
- **Filtered** — semantic search + metadata type filter from classifier output
- **Hybrid (recommended)** — retrieve 20 candidates, re-rank with 60% semantic + 40% type-match scoring

For a detailed walkthrough of exactly what happens at each stage (with a traced example), see **[PIPELINE.md](PIPELINE.md)**.

## Project Structure

```
DSA-LLM/
├── notebooks/
│   ├── 01_explore_data.ipynb           # Data exploration & EDA
│   ├── 02_embeddings.ipynb             # Embeddings + ChromaDB indexing
│   ├── 03_query_pipeline.ipynb         # Basic RAG pipeline with Llama 3.2
│   ├── 04_classification.ipynb         # Multi-label classifier training
│   └── 05_rag_with_classification.ipynb # Enhanced RAG with classification
├── data/
│   ├── homeless_services_hackathon.json # 1,719 services (33 fields each)
│   ├── chroma_db/                       # ChromaDB vector database
│   └── Data Dictionary_*.pdf            # Field documentation
├── models/                              # Trained models (not tracked in git)
│   ├── service_type_classifier_v2_final/  # Production multi-label classifier
│   │   ├── model.safetensors
│   │   ├── label_mappings.json
│   │   └── optimal_thresholds.json
│   └── ...                              # Earlier model versions + checkpoints
├── .gitignore
├── README.md
└── PIPELINE.md                      # Detailed pipeline explanation
```

## Notebooks

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `explore_data.ipynb` | Loads and explores the 1,719 services dataset — field coverage, category distributions, text lengths |
| 02 | `embeddings.ipynb` | Encodes services with `all-MiniLM-L6-v2` (384-dim), stores in ChromaDB with metadata |
| 03 | `query_pipeline.ipynb` | Basic RAG: semantic search + Llama 3.2 via Ollama for case manager responses |
| 04 | `classification.ipynb` | Trains a DistilBERT multi-label classifier for 22 service types with class-weighted BCE loss and per-label threshold optimization |
| 05 | `rag_with_classification.ipynb` | Integrates classifier into RAG: query intent detection, hybrid retrieval, intent-aware LLM prompting |

## Service Types (22 Labels)

The classifier predicts which of these types apply to a service or query:

| Category | Types |
|----------|-------|
| **Housing** | Emergency Shelter & Crisis Intervention, Homelessness Prevention & Diversion, Housing Financial Assistance, Transitional & Supportive Housing, Housing Search & Navigation, Home Accessibility & Improvement, Moving & Relocation Support |
| **Support** | Case Management & Coordination, Legal Assistance & Tenant Advocacy, Housing Education & Counseling, Utility & Energy Assistance |
| **Populations** | Family Services, TAY Services, Veteran Services, Senior Services, Refugee Services, Disability Services, Domestic Violence Support |
| **Health & Basic Needs** | Food & Basic Needs Assistance, Mental Health Services, Substance Abuse Disorder |
| **Other** | Other |

## Models & Performance

### Multi-Label Classifier (v2)

| Metric | Default (0.5) | Optimized Thresholds |
|--------|--------------|---------------------|
| F1 Micro | 0.787 | **0.833** |
| F1 Macro | 0.720 | **0.782** |
| Precision | 0.727 | **0.818** |
| Recall | 0.858 | 0.849 |
| Hamming Loss | 0.058 | **0.043** |

Key improvements over v1:
- **Class-weighted BCE loss** — rare classes (Refugee Services, Moving & Relocation) get up to 50x loss weight
- **Per-label threshold optimization** — each label gets its own decision threshold (range: 0.25 - 0.85)
- **Early stopping** — prevents overfitting with patience=3

### Tech Stack

| Component | Tool | Details |
|-----------|------|---------|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim vectors, runs locally |
| Vector DB | ChromaDB | Persistent storage, HNSW indexing |
| Classifier | DistilBERT | 66M params, multi-label, ~10-20ms inference |
| LLM | Llama 3.2 via Ollama | Local inference, no API keys |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed with Llama 3.2 pulled

### Install

```bash
git clone https://github.com/your-username/DSA-LLM.git
cd DSA-LLM

pip install sentence-transformers chromadb transformers datasets \
            accelerate scikit-learn seaborn torch ollama
```

### Pull Llama 3.2

```bash
ollama pull llama3.2
```

### Run

1. Start Ollama in a terminal:
   ```bash
   ollama serve
   ```

2. Run the notebooks in order (01 through 05), or jump to notebook 05 if the ChromaDB database and trained model already exist.

3. In notebook 05, use the unified entry point:
   ```python
   result = ask_case_manager(
       "homeless veteran needs emergency shelter tonight",
       method="hybrid"
   )
   ```

## Data

**Source:** 211 San Diego homeless services database

**1,719 services** with 33 fields including:
- Service name, organization, description
- Eligibility requirements, target populations
- Contact info (phone, address, website)
- Service types, categories, areas of focus
- Hours, capacity, waitlist status, fees

**3 high-level categories:**
- Mental Health and Substance Use Disorder Services (782)
- Food (543)
- Housing and Shelter (394)

**Services have 1-13 type labels** (average 2.8 per service).

## Future Enhancements

| Component | Purpose |
|-----------|---------|
| Information Extraction | Parse messy free-text fields (eligibility, hours) into structured data |
| Client-Specific Ranking | Score services for specific client profiles (age, situation, location) |
| Web Interface | Streamlit/Gradio app for case managers |
