# 🦅 GeorgeHawk — AI-Powered Job Application System

> **Production-grade** personal job hunt automation built on AIHawk.  
> Scrapes → Scores with ML → Generates tailored cover letters → Applies automatically.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: AGPL](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL-3.0)
[![LLM: Groq (Free)](https://img.shields.io/badge/LLM-Groq%20FREE-green.svg)](https://console.groq.com)
[![Status: Production](https://img.shields.io/badge/status-production-brightgreen.svg)]()

---

## Table of Contents

- [What Is GeorgeHawk?](#what-is-georgehawk)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Environment Setup (.env)](#environment-setup)
- [Configuration (config.yaml)](#configuration)
- [Running the System](#running-the-system)
- [Docker Deployment](#docker-deployment)
- [ML Scoring System](#ml-scoring-system)
- [Logging & Monitoring](#logging--monitoring)
- [ATS Optimisation](#ats-optimisation)
- [Visa & Region Strategy](#visa--region-strategy)
- [Rate Limits & Ethics](#rate-limits--ethics)
- [Upgrade Roadmap](#upgrade-roadmap)
- [Troubleshooting](#troubleshooting)

---

## What Is GeorgeHawk?

GeorgeHawk is a personal ML job application pipeline that:

| Layer | What it does |
|-------|-------------|
| **Scraping** | Pulls jobs from RemoteOK (API), Reddit (JSON API), LinkedIn (Selenium) |
| **ML Scoring** | Ranks each job by cosine similarity between your CV and the JD (TF-IDF) |
| **Filtering** | Removes blacklisted companies/titles; detects visa/relocation offers |
| **Generation** | Writes tailored cover letters via Groq LLaMA 3.3 (free tier) |
| **Storage** | Deduplicates + persists to SQLite; MD5 hash prevents re-applying |
| **Application** | Automates LinkedIn Easy Apply via Selenium |
| **Reporting** | Daily ranked `top_jobs.json` + structured session logs |

**Cost: £0/month** (Groq free tier: 14,400 requests/day; RemoteOK & Reddit are free APIs).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GeorgeHawk Pipeline                     │
│                                                             │
│  [RemoteOK API] ──┐                                         │
│  [Reddit API]  ───┼──► [Job Objects] ──► [TF-IDF Scorer]   │
│  [LinkedIn]    ──┘           │                  │           │
│                              ▼                  ▼           │
│                        [Blacklist]        [fit_score]       │
│                        [Dedup MD5]        [+ keyword_boost] │
│                              │                  │           │
│                              └────────┬──────────┘          │
│                                       ▼                     │
│                              [Groq LLaMA 3.3]               │
│                              [Cover Letter Gen]             │
│                                       │                     │
│                                       ▼                     │
│                              [SQLite jobs.db]               │
│                                       │                     │
│                         ┌─────────────┴──────────┐          │
│                         ▼                         ▼          │
│                  [top_jobs.json]        [Selenium Applier]  │
│                  [georgehawk.log]       [LinkedIn EasyApply]│
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone & create environment

```bash
git clone https://github.com/Georginh0/georgehawk.git
cd georgehawk
conda create -n georgehawk python=3.11 -y
conda activate georgehawk
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Open .env and fill in your credentials (see Environment Setup below)
nano .env
```

### 3. Run job scraping + ML scoring (no browser required)

```bash
python main.py --max-jobs 50
```

### 4. Review your top jobs

```bash
cat top_jobs.json
```

### 5. Run LinkedIn Easy Apply automation

```bash
python selenium_applier.py
```

---

## Environment Setup

### `.env.example` vs `.env` — what's the difference?

| File | Purpose | Committed to Git? |
|------|---------|-------------------|
| `.env.example` | Template showing which variables are needed — **no real values** | ✅ Yes — safe to share |
| `.env` | Your actual secrets and credentials | ❌ **Never** — in `.gitignore` |

**Workflow:**

```bash
# Step 1: copy the template
cp .env.example .env

# Step 2: open .env and replace placeholder values with your real ones
nano .env   # or use VS Code, vim, etc.
```

### Variable reference

```bash
# .env — fill these in, then save. Never commit this file.

# Groq API key — get it FREE at https://console.groq.com
# Format: gsk_ followed by a long string
GROQ_API_KEY=gsk_your_actual_key_here

# LinkedIn credentials for Selenium Easy Apply automation
LINKEDIN_EMAIL=your_actual_email@gmail.com
LINKEDIN_PASSWORD=your_linkedin_password

# Your phone number (used in application forms)
PHONE=+2347086276797

# Path to your PDF CV (relative to project root, or absolute path)
CV_PATH=George_Dogo_CV_Updated_2026.pdf
```

> **Security note:** Your `.env` file contains passwords. Never paste it into AI chats, Slack, or GitHub issues. If you accidentally commit it, rotate your passwords immediately and run `git rm --cached .env`.

---

## Configuration

`config.yaml` controls job search behaviour — safe to commit (no secrets).

```yaml
keywords:
  - "data scientist"
  - "ml engineer"
  - "LangGraph"           # high-signal keyword for your profile

locations:
  - "United Kingdom"      # Priority 1 — Skilled Worker Visa
  - "Germany"             # Priority 2 — EU Blue Card
  - "Canada"
  - "Remote"

min_fit_score: 0.05       # Lower = more jobs; raise to 0.10+ once you have volume
max_applications_per_session: 20   # Stay under LinkedIn's soft rate limit

blacklisted_companies:
  - "staffing"            # Avoid agencies
  - "pyramid"

blacklisted_titles:
  - "intern"
  - "junior frontend"
```

**Tuning `min_fit_score`:**

| Value | Effect |
|-------|--------|
| `0.03` | Casts wide net — more noise |
| `0.05` | Recommended default |
| `0.10` | Tighter filter — high precision, lower recall |
| `0.15` | Priority-only mode |

---

## Running the System

### Scrape + score only (no browser, no LLM calls)

```bash
python main.py --max-jobs 100 --no-letters
```

### Scrape + score + generate cover letters

```bash
python main.py --max-jobs 50
```

### LinkedIn Easy Apply automation

```bash
python -c "
from selenium_applier import run_linkedin_session
from main import ApplicationWriter, JobStore

run_linkedin_session(
    email='your_email@gmail.com',
    password='your_password',
    keywords=['data scientist', 'ml engineer'],
    locations=['United Kingdom', 'Germany', 'Canada'],
    max_applications=15,
    writer=ApplicationWriter(),
    store=JobStore()
)
"
```

### Scheduled daily run (cron)

```bash
# Run every morning at 08:00
crontab -e

# Add this line:
0 8 * * * cd /path/to/georgehawk && conda run -n georgehawk python main.py --max-jobs 30 >> /var/log/georgehawk_cron.log 2>&1
```

### Inspect your database

```bash
sqlite3 jobs.db "SELECT title, company, fit_score, applied FROM jobs ORDER BY fit_score DESC LIMIT 20;"
```

---

## Docker Deployment

### Build & run

```bash
docker compose up --build
```

### Run one-off scrape session

```bash
docker compose run --rm georgehawk python main.py --max-jobs 50 --no-letters
```

### Environment variables with Docker

```bash
# docker-compose.yml reads from your .env file automatically
# Make sure .env exists before running docker compose
```

---

## ML Scoring System

### How it works

```
CV Text  ──► TF-IDF vectorizer ──► cv_vector  (shape: 1 × 5000)
JD Text  ──► TF-IDF vectorizer ──► jd_vector  (shape: 1 × 5000)
                                         │
                               cosine_similarity(cv_vector, jd_vector)
                                         │
                                   base_score ∈ [0, 1]
                                         │
                              + keyword_boost (max 0.15)
                                         │
                                   fit_score ∈ [0, 1]
```

### Decision thresholds

| Score | Action |
|-------|--------|
| `< 0.05` | Skip — irrelevant |
| `0.05 – 0.10` | Save to DB — review manually |
| `0.10 – 0.15` | Generate cover letter |
| `≥ 0.15` | Priority — apply immediately |

### Upgrading to semantic similarity (Phase 2)

Replace `JobFitScorer` with sentence-transformers for semantic matching:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB, runs on CPU
cv_embedding = model.encode(CV_TEXT)
jd_embedding = model.encode(job_description)
score = util.cos_sim(cv_embedding, jd_embedding).item()
```

---

## Logging & Monitoring

Logs are written to both stdout and `georgehawk.log` in the project root.

### Log levels

| Level | When used |
|-------|-----------|
| `INFO` | Normal operation — jobs found, letters generated, applications submitted |
| `WARNING` | Non-fatal issues — CV not found, login needs manual check |
| `ERROR` | API failures, Selenium errors — session continues |
| `DEBUG` | Verbose detail — form parsing, individual field fills |

### Enable debug logging

```bash
python main.py --max-jobs 50 --log-level DEBUG
```

### Log rotation (production)

```python
# In logging_config.py (see improvements.py)
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "georgehawk.log",
    maxBytes=5 * 1024 * 1024,   # 5 MB
    backupCount=7               # keep 7 rotated files
)
```

### Session stats

```
============================================================
SESSION COMPLETE
  Jobs in database : 47
  Applied          : 12
  Pending          : 35
  Avg fit score    : 0.089
============================================================
```

---

## ATS Optimisation

ATS systems (Workday, Greenhouse, Lever, Taleo) score your CV before a human sees it.

**George's ATS keyword set (always include in CV summary):**

```
Core:      data scientist · machine learning · Python · scikit-learn · pandas
LLM/AI:    LangGraph · LangChain · LLM · multi-agent · AI agent
MLOps:     MLflow · ZenML · Docker · CI/CD · model deployment
Backend:   FastAPI · REST API · Flask · JWT
Data:      ETL · ELT · Apache Spark · pipeline · SQL · MySQL
Stats:     A/B testing · hypothesis testing · statistical modelling
```

**Generate a tailored CV summary for any job:**

```python
from main import ApplicationWriter, Job

writer = ApplicationWriter()
job = Job(
    title="Senior ML Engineer",
    company="Babylon Health",
    location="London",
    url="",
    description="...paste JD here..."
)
print(writer.tailor_cv_summary(job))
```

---

## Visa & Region Strategy

| Region | Visa Route | George's Status | Priority |
|--------|-----------|-----------------|----------|
| 🇬🇧 UK | Skilled Worker Visa | Data Science on shortage list | 1 |
| 🇩🇪 Germany | EU Blue Card (€43,800+ threshold) | Qualifies with salary | 2 |
| 🇨🇦 Canada | Express Entry / LMIA | Eligible | 3 |
| 🇦🇺 Australia | 482 Employer Sponsored | MLTSSL eligible | 4 |
| 🇦🇪 UAE | Company-arranged (easy) | No tax, fast process | 5 |

**LinkedIn search hack:** Append `"visa sponsorship"` to your keyword string.  
**Germany resource:** [make-it-in-germany.com](https://www.make-it-in-germany.com)

---

## Rate Limits & Ethics

| Platform | Safe daily limit | GeorgeHawk default |
|----------|-----------------|-------------------|
| LinkedIn Easy Apply | 20–30/day | 20 |
| Groq API | 14,400 req/day | Well within |
| Reddit API | 60 req/min | ~5 req/min |
| RemoteOK | Polite: 1 req/session | 1 req/session |

GeorgeHawk adds `random.uniform(10, 20)` second pauses between LinkedIn applications to mimic human behaviour. **Do not remove these.**

---

## Upgrade Roadmap

### Phase 1 — Current (complete)
- [x] TF-IDF cosine similarity scoring
- [x] Multi-platform scraping (RemoteOK, Reddit, LinkedIn)
- [x] Groq LLaMA cover letter generation
- [x] SQLite deduplication
- [x] Selenium Easy Apply automation

### Phase 2 — With outcome data (50+ applications)
- [ ] Label outcomes: `reply=1 / no_reply=0`
- [ ] Train LightGBM reply predictor on `[fit_score, visa_sponsor, remote, salary_mentioned]`
- [ ] Replace threshold filter with `P(reply | features) > 0.3`
- [ ] Add sentence-transformers for semantic similarity

### Phase 3 — Scale
- [ ] PostgreSQL for multi-device sync
- [ ] Celery + Redis task queue for async scraping
- [ ] Prometheus metrics endpoint
- [ ] Grafana dashboard for application funnel analytics
- [ ] Wellfound Selenium scraper

---

## Troubleshooting

### `GROQ_API_KEY not set`
```bash
# Check your .env exists and has the key
cat .env | grep GROQ
# If missing, get a free key at https://console.groq.com
```

### `scikit-learn not installed`
```bash
pip install scikit-learn>=1.4.0
# Scoring will fall back to 0.5 for all jobs without it
```

### LinkedIn asks for CAPTCHA / 2FA
```bash
# Run with headless=False (default) so you can complete the challenge manually
# GeorgeHawk will wait at the login step
```

### `CV not found at George_Dogo_CV_Updated_2026.pdf`
```bash
# Update CV_PATH in .env to the absolute path of your CV
CV_PATH=/home/george/documents/George_Dogo_CV_Updated_2026.pdf
```

### `sqlite3.OperationalError: database is locked`
```bash
# Another process is using jobs.db — close it or use a different session
```

### All jobs scoring 0.5
```bash
# scikit-learn not installed — scorer falls back to default
pip install scikit-learn numpy
```

---

## Contributing

This is a personal project. If you fork it, update `GEORGE_SUMMARY` in `main.py`
and `GEORGE_ANSWERS` in `selenium_applier.py` with your own background.

---

## License

Based on [AIHawk](https://github.com/feder-cr/Jobs_Applier_AI_Agent) (AGPL-3.0).  
All modifications by George Dogo, 2025–2026.

---

*GeorgeHawk v1.0 · Built by George Dogo · [github.com/Georginh0](https://github.com/Georginh0)*
