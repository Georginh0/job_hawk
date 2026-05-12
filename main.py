"""
GeorgeHawk — Personal AI Job Applier
Built on AIHawk architecture, rebuilt for George Dogo's job search.

Platforms: LinkedIn · Wellfound · Reddit (r/forhire, r/remotework) · RemoteOK
LLM:       Groq llama-3.3-70b (FREE — no OpenAI cost)
Scoring:   TF-IDF ML job-fit scoring against your CV
Target:    Europe · Canada · Australia · Brazil · Gulf nations
"""

import os, re, time, json, logging, hashlib, yaml, random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("georgehawk.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("GeorgeHawk")


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────
@dataclass
class Job:
    title: str
    company: str
    location: str
    url: str
    description: str = ""
    source: str = ""
    salary: str = ""
    remote: bool = False
    visa_sponsor: bool = False
    relocation: bool = False
    fit_score: float = 0.0
    applied: bool = False
    applied_at: Optional[str] = None
    job_id: str = field(default="")

    def __post_init__(self):
        self.job_id = hashlib.md5(
            f"{self.title}{self.company}{self.url}".encode()
        ).hexdigest()[:12]


@dataclass
class JobSearchConfig:
    keywords: list[str]
    locations: list[str]
    remote_only: bool
    require_visa_sponsor: bool
    require_relocation: bool
    min_fit_score: float
    blacklisted_companies: list[str]
    blacklisted_titles: list[str]
    experience_level: list[str]


# ─────────────────────────────────────────────
# GROQ LLM CLIENT (free)
# ─────────────────────────────────────────────
class GroqLLM:
    """
    Wraps Groq API. Free tier: 14,400 requests/day.
    Model: llama-3.3-70b-versatile — smarter than GPT-3.5, free.
    """
    BASE = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in .env")

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        try:
            r = requests.post(self.BASE, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error(f"Groq API error: {e}")
            return ""


# ─────────────────────────────────────────────
# ML JOB FIT SCORER (TF-IDF cosine similarity)
# ─────────────────────────────────────────────
class JobFitScorer:
    """
    Scores job fit against George's CV using TF-IDF cosine similarity.
    No API calls needed — pure sklearn. Fast and free.
    """
    CV_TEXT = """
    Data Scientist AI Engineer LangGraph multi-agent systems LLMOps machine learning
    ETL ELT pipelines Python pandas numpy scikit-learn FastAPI REST API JWT authentication
    signal processing Butterworth FFT IMU accelerometer gyroscope Random Forest XGBoost
    gradient boosting SVM classification clustering neural networks TensorFlow PyTorch
    Apache Spark Kafka Dask ZenML MLflow Docker CI/CD MySQL SQL NoSQL
    Power BI Tableau Streamlit Matplotlib Seaborn Plotly
    A/B testing hypothesis testing Bayesian inference regression anomaly detection
    biosensor wearables health tech dental AI healthcare automation
    distributed computing 50M records pipeline optimisation 30% runtime reduction
    4 years experience Nigeria Lagos Abuja relocation remote
    """

    def __init__(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000
            )
            self.cv_vec = None
            self._fit()
        except ImportError:
            log.warning("scikit-learn not installed — fit scoring disabled")
            self.vectorizer = None

    def _fit(self):
        self.cv_vec = self.vectorizer.fit_transform([self.CV_TEXT])

    def score(self, job_description: str) -> float:
        if not self.vectorizer or not job_description:
            return 0.5
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            jd_vec = self.vectorizer.transform([job_description])
            sim = cosine_similarity(self.cv_vec, jd_vec)[0][0]
            return round(float(sim), 3)
        except Exception:
            return 0.5

    def keyword_boost(self, text: str) -> float:
        """Extra score for high-value keywords for George's profile."""
        boost_keywords = [
            "langgraph", "langchain", "llm", "ai engineer", "ml engineer",
            "data scientist", "machine learning", "signal processing",
            "visa sponsorship", "relocation", "remote",
            "python", "fastapi", "streamlit", "groq",
            "health tech", "wearables", "iot", "sensor"
        ]
        text_lower = text.lower()
        hits = sum(1 for kw in boost_keywords if kw in text_lower)
        return min(hits * 0.02, 0.15)  # max 0.15 boost


# ─────────────────────────────────────────────
# JOB SOURCES
# ─────────────────────────────────────────────
class RemoteOKScraper:
    """
    RemoteOK public JSON API — no auth, no cost.
    Returns remote jobs worldwide.
    """
    BASE = "https://remoteok.com/api"

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []
        try:
            headers = {"User-Agent": "GeorgeHawk/1.0 (job search bot)"}
            r = requests.get(self.BASE, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            # First element is legal notice — skip it
            for item in data[1:]:
                title = item.get("position", "")
                if not any(kw.lower() in title.lower() for kw in keywords):
                    continue
                jobs.append(Job(
                    title=title,
                    company=item.get("company", ""),
                    location="Remote",
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                    source="RemoteOK",
                    salary=item.get("salary", ""),
                    remote=True,
                    visa_sponsor=False,
                ))
        except Exception as e:
            log.error(f"RemoteOK error: {e}")
        log.info(f"RemoteOK: {len(jobs)} jobs found")
        return jobs


class WellfoundScraper:
    """
    Wellfound (AngelList) — startup jobs.
    Uses public search endpoint.
    Focus: Europe, Canada, Australia, Gulf startups.
    """
    BASE = "https://wellfound.com/jobs"

    def fetch(self, keywords: list[str], locations: list[str]) -> list[Job]:
        """
        NOTE: Wellfound requires browser automation for full access.
        This returns a structured job list format ready for Selenium fill-in.
        Add your Selenium scraper here using the Job dataclass above.

        For now: returns seed jobs from Wellfound's public RSS/JSON where available.
        """
        jobs = []
        log.info("Wellfound: browser automation needed — see selenium_scraper.py")
        return jobs


class RedditJobScraper:
    """
    Scrapes r/forhire, r/remotework, r/datascience for job posts.
    Uses Reddit public JSON API — no OAuth needed for read-only.
    """
    SUBREDDITS = ["forhire", "remotework", "MachineLearning", "learnmachinelearning"]
    BASE = "https://www.reddit.com/r/{sub}/search.json"

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []
        headers = {"User-Agent": "GeorgeHawk/1.0"}
        for sub in self.SUBREDDITS:
            for kw in keywords[:2]:  # avoid rate limit
                try:
                    url = self.BASE.format(sub=sub)
                    params = {"q": kw, "sort": "new", "limit": 10, "restrict_sr": 1}
                    r = requests.get(url, headers=headers, params=params, timeout=10)
                    if r.status_code != 200:
                        continue
                    for post in r.json()["data"]["children"]:
                        d = post["data"]
                        title = d.get("title", "")
                        if "[hiring]" in title.lower() or "looking for" in title.lower():
                            jobs.append(Job(
                                title=title[:100],
                                company="via Reddit",
                                location="Remote",
                                url=f"https://reddit.com{d.get('permalink','')}",
                                description=d.get("selftext", "")[:500],
                                source=f"Reddit/r/{sub}",
                                remote=True,
                            ))
                    time.sleep(1)  # rate limit respect
                except Exception as e:
                    log.debug(f"Reddit r/{sub} error: {e}")
        log.info(f"Reddit: {len(jobs)} posts found")
        return jobs


class LinkedInJobScraper:
    """
    LinkedIn job search via public URL scraping.
    For full Easy Apply automation, pair with selenium_applier.py.

    Targets:
      - UK (visa sponsorship)
      - Germany (EU Blue Card)
      - Canada (LMIA)
      - Australia (482 sponsor)
      - UAE/Saudi (Gulf)
      - Brazil (remote-friendly)
    """
    SEARCH_URLS = {
        "UK": "https://www.linkedin.com/jobs/search/?keywords={kw}&location=United+Kingdom&f_WT=2&f_TPR=r604800",
        "Germany": "https://www.linkedin.com/jobs/search/?keywords={kw}&location=Germany&f_WT=2&f_TPR=r604800",
        "Canada": "https://www.linkedin.com/jobs/search/?keywords={kw}&location=Canada&f_WT=2&f_TPR=r604800",
        "Australia": "https://www.linkedin.com/jobs/search/?keywords={kw}&location=Australia&f_WT=2&f_TPR=r604800",
        "UAE": "https://www.linkedin.com/jobs/search/?keywords={kw}&location=United+Arab+Emirates&f_TPR=r604800",
        "Remote": "https://www.linkedin.com/jobs/search/?keywords={kw}&f_WT=2&f_TPR=r604800",
    }

    def get_search_urls(self, keywords: list[str]) -> dict:
        """Returns formatted search URLs per region."""
        return {
            region: url.format(kw="+".join(keywords[:3]))
            for region, url in self.SEARCH_URLS.items()
        }


# ─────────────────────────────────────────────
# COVER LETTER & ANSWER GENERATOR
# ─────────────────────────────────────────────
class ApplicationWriter:
    """
    Uses Groq LLaMA to generate tailored cover letters and
    answer application form questions — specific to George's profile.
    """
    GEORGE_SUMMARY = """
    George Dogo is a Data Scientist and AI Engineer with 4+ years of experience.
    He built DentAI Pro: a production LangGraph multi-agent system for dental clinics
    with 7 specialised nodes, atomic MySQL transactions, FastAPI REST layer, and
    Streamlit UI — deployed at zero cost on Render using Groq LLaMA 3.3.
    He also built a Human Activity Recognition ML pipeline achieving 94% accuracy
    on 6-axis wrist IMU data using Butterworth filtering, FFT features, and Random Forest —
    replicating what Apple Watch and Garmin do internally.
    Skills: Python, LangGraph, FastAPI, scikit-learn, XGBoost, SciPy, MLflow, ZenML,
    Apache Spark, MySQL, Power BI, Streamlit.
    He is open to relocation to Europe, Canada, Australia, Gulf nations, Brazil.
    He holds a B.Sc. Aerospace Engineering from Kingston University London.
    GitHub: github.com/Georginh0
    """

    def __init__(self):
        self.llm = GroqLLM()

    def cover_letter(self, job: Job) -> str:
        system = (
            "You are George Dogo's professional writing assistant. "
            "Write concise, specific, ATS-friendly cover letters. "
            "Never use generic phrases. Always reference the specific company and role. "
            "Maximum 250 words. No 'Dear Sir/Madam'."
        )
        user = f"""
Write a cover letter for George applying to this role:

Job Title: {job.title}
Company: {job.company}
Location: {job.location}
Description: {job.description[:600]}

About George:
{self.GEORGE_SUMMARY}

Requirements:
- Open with a specific hook referencing the company/role
- Mention DentAI Pro or the fitness tracker where relevant
- Close with relocation/remote flexibility
- ATS keywords: match the job description language
- 200-250 words maximum
"""
        return self.llm.complete(system, user, max_tokens=400)

    def answer_question(self, question: str, job: Job) -> str:
        system = (
            "You are answering job application questions on behalf of George Dogo. "
            "Be specific, honest, and concise. Use first person. 2-3 sentences max per answer."
        )
        user = f"""
Question: {question}
Job: {job.title} at {job.company}
George's background: {self.GEORGE_SUMMARY[:400]}

Answer naturally and specifically.
"""
        return self.llm.complete(system, user, max_tokens=200)

    def tailor_cv_summary(self, job: Job) -> str:
        system = (
            "You are rewriting the professional summary section of George Dogo's CV "
            "to precisely match this job description. Keep it under 80 words. "
            "Use keywords from the job description. Do not invent experience."
        )
        user = f"""
Job: {job.title} at {job.company}
JD excerpt: {job.description[:500]}

George's base summary:
Data Scientist and AI Engineer with 4+ years building production ML systems,
LangGraph multi-agent architectures, ETL/ELT pipelines, and signal processing models.
Built DentAI Pro (LangGraph+Groq+MySQL+FastAPI) and a HAR ML system (94% accuracy,
Butterworth+FFT+Random Forest). Open to relocation. github.com/Georginh0

Rewrite for ATS match:
"""
        return self.llm.complete(system, user, max_tokens=200)


# ─────────────────────────────────────────────
# DEDUPLICATION + STORAGE
# ─────────────────────────────────────────────
class JobStore:
    """SQLite-backed job store with deduplication."""

    def __init__(self, db_path: str = "jobs.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT, company TEXT, location TEXT, url TEXT,
                source TEXT, salary TEXT, remote INTEGER,
                visa_sponsor INTEGER, relocation INTEGER,
                fit_score REAL, applied INTEGER DEFAULT 0,
                applied_at TEXT, description TEXT,
                cover_letter TEXT, created_at TEXT
            )
        """)
        self.conn.commit()

    def exists(self, job_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,))
        return cur.fetchone() is not None

    def save(self, job: Job, cover_letter: str = ""):
        if self.exists(job.job_id):
            return
        self.conn.execute("""
            INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            job.job_id, job.title, job.company, job.location, job.url,
            job.source, job.salary, int(job.remote),
            int(job.visa_sponsor), int(job.relocation),
            job.fit_score, int(job.applied), job.applied_at,
            job.description[:1000], cover_letter,
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def mark_applied(self, job_id: str):
        self.conn.execute(
            "UPDATE jobs SET applied=1, applied_at=? WHERE job_id=?",
            (datetime.now().isoformat(), job_id)
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        cur = self.conn.execute("""
            SELECT
                COUNT(*) total,
                SUM(applied) applied,
                COUNT(*) - SUM(applied) pending,
                AVG(fit_score) avg_score
            FROM jobs
        """)
        row = cur.fetchone()
        return {"total": row[0], "applied": row[1], "pending": row[2], "avg_score": round(row[3] or 0, 3)}

    def top_jobs(self, n: int = 10) -> list:
        cur = self.conn.execute(
            "SELECT title, company, location, fit_score, url FROM jobs WHERE applied=0 ORDER BY fit_score DESC LIMIT ?",
            (n,)
        )
        return cur.fetchall()


# ─────────────────────────────────────────────
# VISA / RELOCATION FILTER
# ─────────────────────────────────────────────
VISA_KEYWORDS = [
    "visa sponsorship", "visa sponsor", "will sponsor", "sponsorship provided",
    "skilled worker visa", "tier 2", "lmia", "eu blue card", "482 visa",
    "relocation package", "relocation assistance", "we relocate", "relocation provided",
    "open to relocation", "remote worldwide", "remote global"
]

def check_visa_relocation(text: str) -> tuple[bool, bool]:
    t = text.lower()
    visa = any(kw in t for kw in VISA_KEYWORDS[:8])
    reloc = any(kw in t for kw in VISA_KEYWORDS[8:])
    return visa, reloc


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────
class GeorgeHawk:
    """
    Main job hunt orchestrator.
    Scrapes → Scores → Filters → Writes → Stores → Reports.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.scorer = JobFitScorer()
        self.writer = ApplicationWriter()
        self.store = JobStore()
        self.scrapers = {
            "remoteok": RemoteOKScraper(),
            "reddit": RedditJobScraper(),
            "linkedin": LinkedInJobScraper(),
        }
        log.info("GeorgeHawk initialised ✅")

    def _load_config(self, path: str) -> JobSearchConfig:
        if not Path(path).exists():
            log.warning(f"{path} not found — using defaults")
            return JobSearchConfig(
                keywords=["data scientist", "ml engineer", "ai engineer", "LangGraph"],
                locations=["UK", "Germany", "Canada", "Australia", "UAE", "Remote"],
                remote_only=False,
                require_visa_sponsor=False,
                require_relocation=False,
                min_fit_score=0.05,
                blacklisted_companies=[],
                blacklisted_titles=["intern", "unpaid", "junior frontend"],
                experience_level=["mid", "senior"]
            )
        with open(path) as f:
            d = yaml.safe_load(f)
        return JobSearchConfig(**d)

    def _is_blacklisted(self, job: Job) -> bool:
        title_lower = job.title.lower()
        company_lower = job.company.lower()
        if any(bl.lower() in title_lower for bl in self.config.blacklisted_titles):
            return True
        if any(bl.lower() in company_lower for bl in self.config.blacklisted_companies):
            return True
        return False

    def run(self, max_jobs: int = 50, generate_letters: bool = True):
        log.info("=" * 60)
        log.info("GeorgeHawk — Job Hunt Session Starting")
        log.info(f"Target regions: {self.config.locations}")
        log.info(f"Keywords: {self.config.keywords}")
        log.info("=" * 60)

        all_jobs: list[Job] = []

        # ── Scrape ──────────────────────────────────────────
        all_jobs += self.scrapers["remoteok"].fetch(self.config.keywords)
        all_jobs += self.scrapers["reddit"].fetch(self.config.keywords)

        log.info(f"Total raw jobs collected: {len(all_jobs)}")

        # ── Score + Filter ──────────────────────────────────
        processed = 0
        for job in all_jobs:
            if self.store.exists(job.job_id):
                continue
            if self._is_blacklisted(job):
                log.debug(f"Blacklisted: {job.title} @ {job.company}")
                continue

            # ML fit score
            base_score = self.scorer.score(job.description)
            boost = self.scorer.keyword_boost(f"{job.title} {job.description}")
            job.fit_score = min(base_score + boost, 1.0)

            if job.fit_score < self.config.min_fit_score:
                continue

            # Visa / relocation detection
            job.visa_sponsor, job.relocation = check_visa_relocation(
                f"{job.title} {job.description}"
            )

            # Generate cover letter for good fits
            cover = ""
            if generate_letters and job.fit_score > 0.1:
                cover = self.writer.cover_letter(job)
                log.info(f"✍️  Cover letter generated: {job.title} @ {job.company} (score: {job.fit_score:.2f})")

            self.store.save(job, cover)
            processed += 1

            if processed >= max_jobs:
                break

        # ── Report ──────────────────────────────────────────
        stats = self.store.get_stats()
        log.info("\n" + "=" * 60)
        log.info("SESSION COMPLETE")
        log.info(f"  Jobs in database : {stats['total']}")
        log.info(f"  Applied          : {stats['applied']}")
        log.info(f"  Pending          : {stats['pending']}")
        log.info(f"  Avg fit score    : {stats['avg_score']}")
        log.info("\nTOP JOBS TO APPLY:")
        for row in self.store.top_jobs(10):
            title, company, loc, score, url = row
            log.info(f"  [{score:.2f}] {title} @ {company} ({loc})")
            log.info(f"         {url}")
        log.info("=" * 60)

        # Save top jobs to JSON for review
        top = self.store.top_jobs(20)
        with open("top_jobs.json", "w") as f:
            json.dump([
                {"title": r[0], "company": r[1], "location": r[2],
                 "fit_score": r[3], "url": r[4]}
                for r in top
            ], f, indent=2)
        log.info("Top jobs saved to top_jobs.json")

        return stats


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GeorgeHawk — AI Job Hunter")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-jobs", type=int, default=50, help="Max jobs per session")
    parser.add_argument("--no-letters", action="store_true", help="Skip cover letter generation")
    args = parser.parse_args()

    hawk = GeorgeHawk(config_path=args.config)
    hawk.run(max_jobs=args.max_jobs, generate_letters=not args.no_letters)
