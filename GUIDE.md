# 🦅 GeorgeHawk — ML System Design Guide
### AI-Powered Job Application System
**Author:** George Dogo · Data Scientist & AI Engineer
**Based on:** AIHawk (feder-cr) — rebuilt with senior-level improvements
**GitHub:** github.com/Georginh0

---

## Table of Contents
1. [System Overview](#system-overview)
2. [ML System Design — Full Breakdown](#ml-system-design)
3. [Architecture Diagram](#architecture)
4. [Component Deep Dive](#components)
5. [Improvements Over Original AIHawk](#improvements)
6. [Setup & Run Guide](#setup)
7. [Target Regions & Visa Strategy](#visa-strategy)
8. [ATS Optimisation Guide](#ats)
9. [Job Hunt Metrics & Tracking](#metrics)
10. [Interview Prep by Region](#interview-prep)

---

## 1. System Overview

GeorgeHawk is a personal AI job application system built on the AIHawk open-source framework,
rebuilt and extended with production ML engineering practices.

**What it does:**
- Scrapes jobs from LinkedIn, RemoteOK, Reddit, and Wellfound
- Scores each job using TF-IDF cosine similarity against your CV (ML fit scoring)
- Detects visa sponsorship and relocation signals automatically
- Generates tailored cover letters using Groq LLaMA 3.3 (free)
- Automates LinkedIn Easy Apply via Selenium
- Stores everything in SQLite with deduplication
- Produces daily ranked job lists and application analytics

**What it does NOT do:**
- It does not guarantee job offers
- It does not fabricate experience — all answers reflect your real background
- It does not spam — rate-limited to 20 applications/day maximum

---

## 2. ML System Design — Full Breakdown

### Problem Framing

This is a **recommendation + automation** system. The ML problem:

```
Input:  Job description (text) + George's CV (text)
Output: Relevance score ∈ [0, 1] + application decision (apply / skip)
```

It is an **information retrieval** problem using **unsupervised similarity scoring**.

### Feature Engineering

**Text representation:**
- Corpus: Job title + description + company name
- Method: TF-IDF (Term Frequency × Inverse Document Frequency)
- N-gram range: (1, 2) — unigrams and bigrams
- Max features: 5,000 terms
- Stop words: English removed

**Why TF-IDF over embeddings?**
- Zero API cost — no OpenAI embeddings call
- Runs offline on your laptop
- Fast enough for 50–500 jobs per session
- Interpretable — you can see which terms drive the score
- Upgrade path: swap in sentence-transformers for semantic similarity

**CV document (the query):**
```
George's CV → concatenated text of title, skills, experience, projects
→ TF-IDF vector of shape (1, 5000)
```

**Job document (the candidate):**
```
JD title + description → TF-IDF vector of shape (1, 5000)
```

**Similarity:**
```
fit_score = cosine_similarity(cv_vector, jd_vector) ∈ [0, 1]
```

**Keyword boost:**
```
boost = Σ (high_value_keywords_present) × 0.02, capped at 0.15
final_score = min(fit_score + boost, 1.0)
```

High-value keywords for George: LangGraph, LLM, FastAPI, signal processing,
wearables, health tech, visa sponsorship, relocation.

### Decision Logic

```
IF final_score < 0.05  → SKIP (irrelevant)
IF company BLACKLISTED → SKIP
IF title BLACKLISTED   → SKIP
IF final_score >= 0.05 → SAVE + generate cover letter
IF final_score >= 0.15 → PRIORITY: apply immediately
```

### Upgrade Path (Phase 2)

Once you have 50+ applications with outcomes (reply/no reply):

```
Label: reply=1, no_reply=0
Features: fit_score, visa_sponsor, relocation, salary_mentioned, remote,
          title_match_score, company_size_estimate
Model: LogisticRegression or LightGBM
Goal: predict P(reply | job_features) → smarter prioritisation
```

This turns GeorgeHawk from a **retrieval system** into a **predictive system**.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GeorgeHawk System                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DATA INGESTION LAYER                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────┐  │
│  │  RemoteOK   │  │  LinkedIn   │  │  Reddit  │  │Wellfound │  │
│  │  JSON API   │  │  Selenium   │  │ JSON API │  │ Selenium │  │
│  └──────┬──────┘  └──────┬──────┘  └────┬─────┘  └────┬─────┘  │
│         │                │              │              │        │
│         └────────────────┴──────────────┴──────────────┘        │
│                               │                                  │
│                      ┌────────▼────────┐                         │
│                      │   Job Objects   │                         │
│                      │ (dataclass)     │                         │
│                      └────────┬────────┘                         │
│                               │                                  │
│  ML SCORING LAYER             │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐   │
│  │  JobFitScorer                                             │   │
│  │  ┌─────────────────┐    ┌──────────────────────────────┐  │   │
│  │  │  TF-IDF Vectorizer│  │  Cosine Similarity           │  │   │
│  │  │  George's CV    │  → │  cos(cv_vec, jd_vec) → score │  │   │
│  │  │  as query doc   │    │  + keyword_boost()           │  │   │
│  │  └─────────────────┘    └──────────────────────────────┘  │   │
│  └────────────────────────────┬──────────────────────────────┘   │
│                               │                                  │
│  FILTERING LAYER              │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐   │
│  │  • Blacklist filter (company, title)                      │   │
│  │  • Deduplication (MD5 hash of title+company+url)          │   │
│  │  • Visa/relocation keyword detection                      │   │
│  │  • Min fit score threshold                                │   │
│  └────────────────────────────┬──────────────────────────────┘   │
│                               │                                  │
│  GENERATION LAYER (Groq LLM)  │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐   │
│  │  ApplicationWriter (llama-3.3-70b-versatile, FREE)        │   │
│  │  • cover_letter(job) → tailored 250-word letter           │   │
│  │  • answer_question(q, job) → specific form answers        │   │
│  │  • tailor_cv_summary(job) → ATS-matched summary           │   │
│  └────────────────────────────┬──────────────────────────────┘   │
│                               │                                  │
│  STORAGE LAYER                │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐   │
│  │  SQLite (jobs.db)                                         │   │
│  │  • Deduplication by job_id (MD5 hash)                     │   │
│  │  • Tracks: title, company, location, score, applied,      │   │
│  │    applied_at, cover_letter, visa_sponsor, relocation     │   │
│  └────────────────────────────┬──────────────────────────────┘   │
│                               │                                  │
│  APPLICATION LAYER            │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐   │
│  │  EasyApplyHandler (Selenium)                              │   │
│  │  • Auto-fill forms with George's pre-defined answers      │   │
│  │  • LLM answers for unknown questions                      │   │
│  │  • CV upload (PDF)                                        │   │
│  │  • Human-like delays (random 10–20s between applies)      │   │
│  └────────────────────────────┬──────────────────────────────┘   │
│                               │                                  │
│  REPORTING LAYER              │                                  │
│  ┌────────────────────────────▼──────────────────────────────┐   │
│  │  • georgehawk.log — full session log                      │   │
│  │  • top_jobs.json — ranked jobs ready to apply             │   │
│  │  • SQLite stats: total/applied/pending/avg_score          │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Deep Dive

### main.py — GeorgeHawk Orchestrator
**Design pattern:** Facade + Strategy
- Coordinates all scrapers, scorer, writer, and store
- Config-driven via YAML (no hardcoded values)
- Stateless between sessions (SQLite handles state)

### JobFitScorer — ML Fit Engine
**Design pattern:** Strategy (swappable similarity model)
- TF-IDF baseline (current)
- Swap to sentence-transformers for semantic similarity (upgrade)
- Keyword boost layer adds domain-specific signal
- Runs on CPU, no GPU needed

### GroqLLM — Language Model Client
**Design pattern:** Adapter
- Wraps Groq REST API in a clean interface
- Low temperature (0.2) for reliable, consistent output
- Retry logic can be added with `tenacity` library
- Free: 14,400 requests/day on Groq free tier

### JobStore — Persistence Layer
**Design pattern:** Repository
- SQLite for simplicity (no database server needed)
- MD5 hash deduplication prevents re-applying to same job
- `top_jobs()` returns ranked list ordered by fit_score DESC
- Upgrade path: PostgreSQL for multi-device sync

### EasyApplyHandler — Application Automation
**Design pattern:** Template Method
- fill_modal() handles one step at a time
- Pre-defined answer dictionary for 90% of common fields
- LLM fallback for unknown questions
- Rate limiting: random 10–20s human-like pauses

### selenium_applier.py — Browser Automation
**Design pattern:** Page Object Model (lightweight)
- Stealth mode: removes webdriver flag, uses real user-agent
- Separate login, search, apply functions
- Headless mode available once stable
- Anti-detection: random delays, realistic scroll patterns

---

## 5. Improvements Over Original AIHawk

| Feature | Original AIHawk | GeorgeHawk |
|---------|-----------------|------------|
| LLM | OpenAI GPT-4o (paid) | Groq LLaMA 3.3 (FREE) |
| Job scoring | None — applies blindly | TF-IDF ML cosine similarity |
| Platforms | LinkedIn only | LinkedIn + RemoteOK + Reddit + Wellfound |
| Deduplication | Basic URL check | MD5 hash of title+company+url |
| Visa detection | None | Regex keyword detection |
| Storage | CSV files | SQLite with full schema |
| Config | YAML | YAML + dataclass validation |
| Logging | Basic print | Structured logging to file + stdout |
| Architecture | Monolithic script | Modular (scraper, scorer, writer, store, applier) |
| CV tailoring | Generic | Per-job summary rewrite via LLM |
| Rate limiting | None | Human-like random delays |
| Cost | £20+/month OpenAI | £0 (Groq free tier) |

---

## 6. Setup & Run Guide

### Step 1 — Clone and install

```bash
git clone https://github.com/Georginh0/georgehawk
cd georgehawk
conda create -n georgehawk python=3.11 -y
conda activate georgehawk
pip install -r requirements.txt
```

### Step 2 — Configure

```bash
cp .env.example .env
# Edit .env:
#   GROQ_API_KEY=gsk_your_key   (from console.groq.com)
#   LINKEDIN_EMAIL=your_email
#   LINKEDIN_PASSWORD=your_password
```

### Step 3 — Run job scraping and scoring (no browser needed)

```bash
python main.py --max-jobs 50
```

Output: `top_jobs.json` — ranked list of best-fit jobs.

### Step 4 — Run LinkedIn Easy Apply automation

```bash
python -c "
from selenium_applier import run_linkedin_session
run_linkedin_session(
    email='George_dogo@aol.com',
    password='your_password',
    keywords=['data scientist', 'ml engineer'],
    locations=['United Kingdom', 'Germany', 'Canada'],
    max_applications=15
)
"
```

### Step 5 — Run daily (cron)

```bash
# Add to crontab: run every morning at 8am
0 8 * * * cd /path/to/georgehawk && conda run -n georgehawk python main.py --max-jobs 30
```

---

## 7. Target Regions & Visa Strategy

### United Kingdom — Priority 1
- **Visa:** Skilled Worker Visa (sponsored by employer)
- **Why George qualifies:** Data science is on the Shortage Occupation List
- **Salary floor:** £26,200+ (data science roles typically £40K–£70K)
- **Best companies:** NHS Digital, Babylon Health, Lloyds Bank, HSBC, fintech startups
- **Search filter:** "visa sponsorship" + "data scientist" + United Kingdom
- **LinkedIn keyword:** Add "visa sponsorship" to search string

### Germany — Priority 2
- **Visa:** EU Blue Card (fastest route for non-EU skilled workers)
- **Salary floor:** €43,800/year
- **Language:** English jobs available in Berlin, Munich, Frankfurt
- **Best companies:** SAP, Deutsche Bank, Berlin AI startups, N26, Zalando
- **Resource:** make-it-in-germany.com

### Canada — Priority 3
- **Visa:** Express Entry (Federal Skilled Worker) or LMIA employer sponsorship
- **Best cities:** Toronto, Vancouver, Ottawa, Calgary
- **Best companies:** Shopify, RBC, TD Bank, Thomson Reuters, AI startups
- **Search:** "data scientist LMIA" OR "data scientist Express Entry"

### Australia — Priority 4
- **Visa:** Skilled Independent (189) or employer-sponsored (482)
- **Data science is on:** Medium and Long-term Strategic Skills List (MLTSSL)
- **Best cities:** Sydney, Melbourne, Brisbane
- **Search:** "data scientist 482 visa" OR "data scientist sponsorship"

### UAE / Gulf — Priority 5
- **No income tax.** Salaries AED 15,000–35,000/month
- **Best cities:** Dubai, Abu Dhabi, Riyadh
- **Companies:** ADNOC, Emirates, Etisalat, Al Rajhi Bank, Noon
- **Visa:** Company arranges — easy for skilled workers

### Brazil — Remote only
- **Remote USD roles** targeting Brazilian companies with global clients
- **Platforms:** 99jobs.com, Catho, Gupy — filter "trabalho remoto"

### Wellfound (AngelList) — All regions
- **Best for:** Seed, Series A/B startups globally
- **Filter:** Remote + Data Science + Europe or Canada
- **Profile tip:** Set "Open to relocating" + list all target countries

---

## 8. ATS Optimisation Guide

### How ATS Systems Work

Most companies use ATS (Applicant Tracking Systems) like:
Workday, Greenhouse, Lever, Taleo, Ashby, Jobvite.

They parse your CV and score it against the job description using keyword matching.
If your score is below a threshold, your CV is rejected before a human sees it.

### George's ATS Rules

**Rule 1: Exact keyword matching**
If the JD says "machine learning engineer" — your CV must say "machine learning engineer."
Not "ML engineer." Not "ML/AI." Exact match.

**Rule 2: Skills section is critical**
ATS parsers weight the Skills section heavily. Ensure your skills section contains:
- Every tool mentioned in the JD
- Exact spelling (not abbreviations unless JD uses them)

**Rule 3: Job title alignment**
Put the target job title in your Professional Summary. If applying for "Data Scientist":
"Data Scientist with 4+ years of experience..." — not just "I build ML models."

**Rule 4: No tables, no headers with icons**
Some ATS parsers cannot read text inside tables. Use plain text CV format.
The Word CV provided uses ATS-safe formatting.

**Rule 5: Use the GeorgeHawk tailor function**
```python
from main import ApplicationWriter
from main import Job

writer = ApplicationWriter()
job = Job(title="ML Engineer", company="Babylon Health",
          location="London", url="", description="your JD here")
print(writer.tailor_cv_summary(job))
```
Use this to rewrite your summary for every application.

### George's ATS Keywords (always include)

Core: data scientist, machine learning, Python, scikit-learn, pandas, NumPy
AI/LLM: LangGraph, LangChain, LLM, AI agent, multi-agent
MLOps: MLflow, ZenML, Docker, CI/CD, model deployment
Backend: FastAPI, REST API, Flask
Data: ETL, ELT, Apache Spark, pipeline, SQL, MySQL
Stats: A/B testing, hypothesis testing, statistical modelling

---

## 9. Job Hunt Metrics & Tracking

Track these weekly in your budget tracker:

| Metric | Week 1 Target | Month 1 Target |
|--------|---------------|----------------|
| Applications sent | 20 | 80 |
| LinkedIn connections | 15 | 60 |
| Recruiter messages received | 2 | 10 |
| Interview invitations | 0–1 | 3–5 |
| Demo calls / technical screens | 0 | 1–2 |
| Offers | 0 | 0–1 |

**Conversion benchmarks (realistic for international applications):**
- Application → Response: 5–10%
- Response → Interview: 40–60%
- Interview → Offer: 20–30%
- Therefore: 80 applications → 4–8 responses → 2–5 interviews → 0–1 offers/month

**Your edge:**
- DentAI Pro with live URL (most candidates have no deployed system)
- Fitness tracker with documented signal processing (rare in Nigeria-based applicants)
- LangGraph experience (extremely in-demand, less than 2% of data scientists have it)

---

## 10. Interview Prep by Region

### UK Technical Interview (NHS Digital, Babylon, Lloyds)
Expect: System design + ML fundamentals + Python coding
- "Design a patient risk scoring system" → talk about your DentAI architecture
- "Explain Random Forest vs XGBoost" → you have both in production projects
- "What is your experience with LangGraph?" → walk through DentAI node structure

### Germany / Netherlands (SAP, N26, Zalando)
Expect: Structured technical assessment + portfolio review
- Prepare a 5-minute demo of DentAI Pro
- Expect questions on distributed systems (Apache Spark experience)
- Emphasise ETL/ELT pipeline work at UXC

### Canada (Shopify, RBC)
Expect: Behavioural + technical mix (STAR format)
- "Tell me about a time you improved a system's performance" → 30% pipeline reduction at UXC
- "Describe your biggest technical challenge" → OHP/Bench Press confusion fix in HAR

### UAE / Gulf (ADNOC, Emirates)
Expect: More relationship-focused, less technical depth in first round
- Emphasise scale (50M+ record datasets)
- Show the DentAI demo — visual demos land well
- Salary: do not disclose first. Ask for their range.

---

## Disclaimer

This tool is for educational purposes. Use responsibly.
Comply with the terms of service of all platforms (LinkedIn, Wellfound, Reddit).
Maximum 20 automated applications per day — stay under platform rate limits.
Always review AI-generated cover letters before sending.

---

*GeorgeHawk v1.0 · Built by George Dogo · github.com/Georginh0*
*Based on AIHawk open-source project (AGPL License)*
