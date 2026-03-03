# How the Pipeline Works

This document traces a real query through every stage of the classification-enhanced RAG pipeline, explaining exactly what happens at each step.

## Overview

The whole system answers one question: **"A case manager types a query — how do we find the best services and give a useful answer?"**

There are 3 stages:

```
User Query
    |
    v
[Stage 1: Classify Intent]  -- DistilBERT detects which of 22 service types the query is about
    |
    v
[Stage 2: Retrieve Services] -- Semantic search + type-match re-ranking
    |
    v
[Stage 3: Generate Response] -- Llama 3.2 produces recommendations with intent context
```

Let's trace a real query through each one.

---

## Example Query: *"homeless veteran needs emergency shelter tonight"*

---

## Stage 1: Classify Intent

**What happens:** The query gets fed into a DistilBERT model (a small, fast version of BERT with 66M parameters) that was fine-tuned on the 1,719 services dataset.

**Mechanically:**
1. The tokenizer breaks the query into tokens: `["homeless", "veteran", "needs", "emergency", "shelter", "tonight"]` (plus special tokens)
2. Each token gets converted to an input ID (a number from DistilBERT's vocabulary)
3. These IDs go through 6 transformer layers that produce a 768-dimensional representation
4. A classification head (a linear layer) maps that 768-dim vector to **22 output numbers** (one per service type)
5. Each number gets passed through **sigmoid** (not softmax!) — this is critical because it means each label is independent. Sigmoid squashes each value to 0-1 independently, so multiple labels can be "on" simultaneously
6. Each probability is compared against that label's **optimized threshold** (not a flat 0.5 — each label has its own threshold tuned to maximize F1). For example, Refugee Services has threshold 0.25 while Legal Assistance has 0.85

**Output for our query:**
```
Emergency Shelter & Crisis Intervention  -> 0.90 (threshold 0.55) -> YES
Transitional & Supportive Housing        -> 0.82 (threshold 0.75) -> YES
Veteran Services                         -> 0.72 (threshold 0.60) -> YES
Food & Basic Needs                       -> 0.40 (threshold 0.35) -> YES
Mental Health Services                   -> 0.30 (threshold 0.40) -> NO
```

**Why this matters:** Now the system *knows* this query is about shelters and veterans, not about food pantries or legal aid. The basic RAG pipeline (notebook 03) didn't know this — it just matched words.

**Key design decisions:**
- **Sigmoid, not softmax**: Softmax forces all 22 probabilities to sum to 1, which would mean predicting "Emergency Shelter" reduces the probability of "Veteran Services." Sigmoid treats each label independently — a query can be about both shelters AND veterans.
- **Per-label thresholds**: Rare classes like Refugee Services (only 11 training samples) need a lower threshold (0.25) to ever get predicted. Common classes like Legal Assistance can afford a higher bar (0.85) to avoid false positives.
- **Threshold scaling for queries (0.7x)**: The classifier was trained on rich service descriptions (~200 tokens). User queries are short (~10 tokens), so confidence is naturally lower. Multiplying all thresholds by 0.7 compensates for this.

---

## Stage 2: Retrieve Services (Hybrid)

This happens in two sub-steps.

### Step 2a: Semantic Search (get candidates)

**What happens:** The query gets encoded into a 384-dimensional vector using `all-MiniLM-L6-v2` (a sentence-transformer model). This is a *different* model than the classifier — it's specifically trained to make semantically similar sentences have similar vectors.

**Mechanically:**
1. The query becomes a 384-dim vector, e.g., `[0.032, -0.118, 0.045, ...]`
2. ChromaDB compares this vector against all 1,719 stored service vectors using L2 (Euclidean) distance
3. Returns the **top 20** closest matches (not 5 — we over-retrieve on purpose)

These 20 candidates are ranked purely by *how similar the meaning is* to the query. A food pantry that mentions "homeless" might rank high even though it's not a shelter.

### Step 2b: Re-rank with Type Matching

**What happens:** We combine the semantic score with a type-match score to re-order those 20 candidates.

**Mechanically, for each of the 20 candidates:**

1. **Semantic score** — convert ChromaDB distance to a 0-1 similarity: `1 / (1 + distance)`, then normalize across all 20 candidates so the best = 1.0 and worst = 0.0

2. **Type-match score** — look at the candidate's `types` metadata (e.g., `"Emergency Shelter & Crisis Intervention, Veteran Services"`). For each type the service has, look up the classifier's probability for that type from Stage 1. Take the **max** probability across matching types. For example:
   - A service tagged "Emergency Shelter, Veteran Services" -> max(0.90, 0.72) = **0.90**
   - A service tagged "Food & Basic Needs" -> max(0.40) = **0.40**
   - A service tagged "Legal Assistance" -> max(0.08) = **0.08**

3. **Combined score** = `0.6 * semantic + 0.4 * type_match`

4. Sort by combined score, return **top 5**

**Why this works:** A service that is semantically similar *and* matches the detected types gets boosted. A food pantry that happened to mention "homeless veterans" might rank #3 by pure semantics, but its type-match score is low (0.40 vs 0.90 for a shelter), so it drops below actual shelters and veteran services.

**Why hybrid instead of just filtering?** If we used the classifier to *filter* (only return services matching detected types), we'd miss services that are relevant but tagged differently. For example, a "Case Management & Coordination" service that specializes in veterans wouldn't match the type filter for "Emergency Shelter," but it might still be useful. Hybrid keeps it in the pool — it just ranks lower.

**Why max-probability scoring (not counting matches)?** If we counted how many types match, a service tagged with 8 types would always outscore one tagged with 2 types, regardless of relevance. Max-probability asks "what's the strongest single type match?" which is a better signal.

---

## Stage 3: Generate Response

**What happens:** The 5 retrieved services + the detected types get packaged into a prompt and sent to Llama 3.2 running locally via Ollama.

**Mechanically, the LLM receives two messages:**

**System prompt** (sets the persona and provides the classification context):
```
You are a helpful assistant for case managers in San Diego.

The client's query has been analyzed and the following service
needs were detected:
Emergency Shelter & Crisis Intervention, Veteran Services, ...

Prioritize services that match these detected needs.
Your job is to: [recommend services, explain eligibility,
provide contact info, note important details...]
```

**User message** (the query + retrieved services as context):
```
A case manager is asking: "homeless veteran needs emergency shelter tonight"

Here are the relevant services from our database:
---
SERVICE 1: Emergency Adult Shelter VVSD
Organization: Veterans Village of San Diego
Phone: (619) 233-8500
Types: Emergency Shelter & Crisis Intervention, Veteran Services
[full description...]
---
SERVICE 2: ...
[4 more services]

Based on these services, provide helpful recommendations.
```

**Why the detected types matter in the prompt:** Without them, the LLM just sees 5 services and a query — it has to guess what's important. With them, it knows "this is about shelters and veterans" and can:
- Prioritize services that match those types
- Call out veteran-specific eligibility requirements
- Flag services that don't match the detected needs
- Structure its response around the identified categories

---

## What Each Model Does

| Model | Type | Input | Output | Purpose |
|-------|------|-------|--------|---------|
| `all-MiniLM-L6-v2` | Sentence Transformer | Text string | 384-dim vector | Encode queries and services into comparable vectors for semantic search |
| `DistilBERT (fine-tuned)` | Multi-label classifier | Text string | 22 probabilities | Detect which service types a query or service is about |
| `Llama 3.2` | Large Language Model | System prompt + context | Natural language | Generate human-readable recommendations from retrieved services |

---

## The Key Insight

Without classification (notebook 03):
```
query -> word matching -> 5 results -> LLM guesses what's relevant
```

With classification (notebook 05):
```
query -> understand WHAT they need -> 20 candidates -> re-rank by relevance -> 5 results -> LLM KNOWS what's relevant
```

The classifier adds *structured understanding* on top of *semantic similarity*. It's the difference between "these words are similar" and "this query is about emergency shelter for veterans."

---

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| Classification (DistilBERT) | ~10-20ms | Small model, runs on CPU |
| ChromaDB query (20 results) | ~5-10ms | HNSW index, all in memory |
| Re-ranking 20 candidates | <1ms | Simple arithmetic |
| LLM generation (Llama 3.2) | ~5-15s | Dominates total latency |

The classification and re-ranking add negligible overhead. The LLM generation step is the bottleneck regardless of retrieval strategy.
