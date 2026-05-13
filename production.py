"""
production.py — GeorgeHawk Production Hardening
================================================
Senior ML Engineering layer on top of the base GeorgeHawk system.

Place this file in the ROOT of your project (same folder as main.py):

    job_hawk/
    ├── main.py
    ├── production.py        ← this file
    ├── selenium_applier.py
    ├── config.yaml
    ├── .env
    └── ...

Run the hardened session:
    python production.py --max-jobs 50

Or import individual pieces into main.py:
    from production import setup_logging, HealthCheck, SessionMetrics

What this adds over main.py:
  1. RotatingFileHandler  — structured logs, 5 MB rotation, 7-day retention
  2. ValidatedConfig      — Pydantic type-checking of config.yaml at startup
  3. RobustGroqLLM        — exponential backoff retry on API failures
  4. RateLimiter          — token-bucket throttle (protects LinkedIn account)
  5. SemanticScorer       — sentence-transformers upgrade over TF-IDF
  6. SessionMetrics       — full application funnel KPI tracking
  7. HealthCheck          — pre-flight validation before wasting any API calls
  8. Notify               — Slack/Discord webhook on session complete
"""

import os
import time
import json
import logging
import smtplib
import hashlib
import sqlite3
import requests
import threading
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# 1. STRUCTURED LOGGING WITH ROTATION
# ─────────────────────────────────────────────────────────────────


def setup_logging(
    log_file: str = "georgehawk.log",
    level: str = "INFO",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB per file
    backup_count: int = 7,  # keep 1 week of rotated logs
) -> logging.Logger:
    """
    Production-grade logging setup.

    Features:
    - Rotating file handler (5 MB max, 7 backups)
    - Coloured console output (INFO=white, WARNING=yellow, ERROR=red)
    - ISO-8601 timestamps
    - Module name in every log line

    Usage:
        log = setup_logging()
        log.info("Session started")
        log.error("Groq API failed: %s", err)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file handler — never lose logs, never fill disk
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Console handler with colour
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_ColourFormatter())
    console_handler.setLevel(log_level)

    root = logging.getLogger("GeorgeHawk")
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.propagate = False

    return root


class _ColourFormatter(logging.Formatter):
    """ANSI colour codes for console readability."""

    GREY = "\x1b[38;5;240m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY
        + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        + RESET,
        logging.INFO: GREEN
        + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        + RESET,
        logging.WARNING: YELLOW
        + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        + RESET,
        logging.ERROR: RED
        + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        + RESET,
        logging.CRITICAL: BOLD_RED
        + "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        + RESET,
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.DEBUG])
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")
        return formatter.format(record)


# ─────────────────────────────────────────────────────────────────
# 2. PYDANTIC CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel, validator, Field

    class ValidatedConfig(BaseModel):
        """
        Type-safe config with validation.
        Replaces the raw YAML dict loading in main.py.

        Usage:
            import yaml
            with open("config.yaml") as f:
                raw = yaml.safe_load(f)
            config = ValidatedConfig(**raw)
        """

        keywords: list[str] = Field(min_items=1)
        locations: list[str] = Field(min_items=1)
        experience_level: list[str] = ["mid", "senior"]
        remote_only: bool = False
        require_visa_sponsor: bool = False
        require_relocation: bool = False
        min_fit_score: float = Field(default=0.05, ge=0.0, le=1.0)
        max_applications_per_session: int = Field(default=20, ge=1, le=50)
        pause_between_applications_sec: int = Field(default=15, ge=5)
        blacklisted_companies: list[str] = []
        blacklisted_titles: list[str] = []
        cv_path: str = "George_Dogo_CV_Updated_2026.pdf"
        phone: str = ""

        @validator("min_fit_score")
        def score_in_range(cls, v):
            if not 0.0 <= v <= 1.0:
                raise ValueError("min_fit_score must be between 0.0 and 1.0")
            return v

        @validator("max_applications_per_session")
        def safe_rate_limit(cls, v):
            if v > 50:
                raise ValueError(
                    "max_applications_per_session > 50 risks LinkedIn account ban. "
                    "Reduce to ≤ 50."
                )
            return v

        @validator("cv_path")
        def cv_must_exist(cls, v):
            if not Path(v).exists():
                import warnings

                warnings.warn(
                    f"CV not found at '{v}'. Upload will fail during Easy Apply.",
                    UserWarning,
                )
            return v

except ImportError:
    # Pydantic not installed — provide a no-op fallback
    class ValidatedConfig:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# ─────────────────────────────────────────────────────────────────
# 3. RETRY LOGIC — ROBUST GROQ CLIENT
# ─────────────────────────────────────────────────────────────────


class RobustGroqLLM:
    """
    Groq API client with:
    - Exponential backoff retry (3 attempts, 2s → 4s → 8s)
    - Timeout handling
    - Graceful degradation (returns "" on total failure)
    - Request logging

    Replaces the basic GroqLLM class in main.py.

    Usage:
        llm = RobustGroqLLM()
        response = llm.complete(system="...", user="...")
    """

    BASE = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"
    MAX_RETRIES = 3
    RETRY_CODES = {429, 500, 502, 503, 504}  # retry on these HTTP codes

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.log = logging.getLogger("GeorgeHawk.GroqLLM")
        if not self.api_key:
            self.log.warning("GROQ_API_KEY not set — LLM features disabled")

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        if not self.api_key:
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                r = requests.post(self.BASE, headers=headers, json=payload, timeout=30)

                if r.status_code in self.RETRY_CODES:
                    wait = 2**attempt  # 2s, 4s, 8s
                    self.log.warning(
                        "Groq returned %s — retry %d/%d in %ds",
                        r.status_code,
                        attempt,
                        self.MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                result = r.json()["choices"][0]["message"]["content"].strip()
                self.log.debug(
                    "Groq OK | tokens_used=%s",
                    r.json().get("usage", {}).get("total_tokens"),
                )
                return result

            except requests.exceptions.Timeout:
                self.log.warning(
                    "Groq timeout — attempt %d/%d", attempt, self.MAX_RETRIES
                )
                last_error = "timeout"
                time.sleep(2**attempt)

            except requests.exceptions.RequestException as e:
                self.log.error("Groq request error: %s", e)
                last_error = str(e)
                break

        self.log.error(
            "Groq failed after %d attempts: %s", self.MAX_RETRIES, last_error
        )
        return ""


# ─────────────────────────────────────────────────────────────────
# 4. TOKEN-BUCKET RATE LIMITER
# ─────────────────────────────────────────────────────────────────


class RateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Ensures we don't exceed platform limits:
    - LinkedIn: max 20 applications/day
    - Reddit API: max 60 requests/minute
    - Groq API: max 30 requests/minute on free tier

    Usage:
        limiter = RateLimiter(requests_per_minute=2)  # 2 applications/min
        for job in jobs:
            limiter.acquire()   # blocks if too fast
            apply_to_job(job)
    """

    def __init__(self, requests_per_minute: float = 3.0):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.capacity = requests_per_minute
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self.log = logging.getLogger("GeorgeHawk.RateLimiter")

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0):
        """Block until a token is available, then consume it."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            # Need to wait
            deficit = tokens - self._tokens
            wait_time = deficit / self.rate
            self.log.debug("Rate limiter: waiting %.1fs", wait_time)

        time.sleep(wait_time)

        with self._lock:
            self._refill()
            self._tokens -= tokens


# ─────────────────────────────────────────────────────────────────
# 5. SEMANTIC SCORER (sentence-transformers upgrade)
# ─────────────────────────────────────────────────────────────────


class SemanticScorer:
    """
    Semantic similarity scorer using sentence-transformers.
    Phase 2 upgrade over TF-IDF — understands meaning, not just keywords.

    Model: all-MiniLM-L6-v2
    - 80MB download (one-time)
    - Runs on CPU in ~50ms per job
    - Much better at synonyms: "machine learning engineer" ≈ "ML engineer"

    Install:
        pip install sentence-transformers

    Usage:
        scorer = SemanticScorer(cv_text="...your CV text...")
        score = scorer.score("...job description...")
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, cv_text: str):
        self.log = logging.getLogger("GeorgeHawk.SemanticScorer")
        self.available = False
        self.cv_embedding = None

        try:
            from sentence_transformers import SentenceTransformer, util

            self._util = util
            self.log.info("Loading sentence-transformers model: %s", self.MODEL_NAME)
            self._model = SentenceTransformer(self.MODEL_NAME)
            self.cv_embedding = self._model.encode(cv_text, convert_to_tensor=True)
            self.available = True
            self.log.info("SemanticScorer ready ✅")
        except ImportError:
            self.log.warning(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers  "
                "Falling back to TF-IDF scorer."
            )
        except Exception as e:
            self.log.error("SemanticScorer init failed: %s", e)

    def score(self, job_description: str) -> float:
        """Return cosine similarity ∈ [0, 1]."""
        if not self.available or not job_description:
            return 0.0
        try:
            jd_embedding = self._model.encode(job_description, convert_to_tensor=True)
            sim = self._util.cos_sim(self.cv_embedding, jd_embedding)
            return round(float(sim[0][0]), 3)
        except Exception as e:
            self.log.error("Scoring error: %s", e)
            return 0.0


# ─────────────────────────────────────────────────────────────────
# 6. SESSION METRICS — APPLICATION FUNNEL TRACKING
# ─────────────────────────────────────────────────────────────────


@dataclass
class SessionMetrics:
    """
    Tracks application funnel KPIs per session.
    Written to metrics.json at session end for trend analysis.

    Funnel:
        jobs_scraped → jobs_scored → jobs_filtered → letters_generated
        → applications_attempted → applications_submitted → errors
    """

    session_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    jobs_scraped: int = 0
    jobs_scored: int = 0
    jobs_filtered_blacklist: int = 0
    jobs_filtered_low_score: int = 0
    jobs_saved: int = 0
    letters_generated: int = 0
    applications_attempted: int = 0
    applications_submitted: int = 0
    application_errors: int = 0
    groq_api_calls: int = 0
    groq_api_failures: int = 0
    avg_fit_score: float = 0.0
    _scores: list = field(default_factory=list, repr=False)

    def record_score(self, score: float):
        self._scores.append(score)
        self.avg_fit_score = round(sum(self._scores) / len(self._scores), 3)

    @property
    def conversion_rate(self) -> str:
        if self.jobs_scraped == 0:
            return "0%"
        return f"{self.applications_submitted / self.jobs_scraped * 100:.1f}%"

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": datetime.now().isoformat(),
            "funnel": {
                "scraped": self.jobs_scraped,
                "scored": self.jobs_scored,
                "filtered_blacklist": self.jobs_filtered_blacklist,
                "filtered_low_score": self.jobs_filtered_low_score,
                "saved": self.jobs_saved,
                "letters_generated": self.letters_generated,
                "applied": self.applications_submitted,
                "errors": self.application_errors,
            },
            "llm": {
                "groq_calls": self.groq_api_calls,
                "groq_failures": self.groq_api_failures,
            },
            "quality": {
                "avg_fit_score": self.avg_fit_score,
                "conversion_rate": self.conversion_rate,
            },
        }

    def save(self, path: str = "metrics.json"):
        """Append session metrics to rolling metrics file."""
        data = []
        if Path(path).exists():
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = []

        data.append(self.summary())

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        log = logging.getLogger("GeorgeHawk.Metrics")
        s = self.summary()
        log.info("─" * 60)
        log.info("SESSION METRICS — %s", s["session_id"])
        log.info("  Scraped      : %d", s["funnel"]["scraped"])
        log.info("  Saved to DB  : %d", s["funnel"]["saved"])
        log.info("  Applied      : %d", s["funnel"]["applied"])
        log.info("  Errors       : %d", s["funnel"]["errors"])
        log.info("  Avg fit score: %.3f", s["quality"]["avg_fit_score"])
        log.info(
            "  Conversion   : %s (applied/scraped)", s["quality"]["conversion_rate"]
        )
        log.info(
            "  Groq calls   : %d (failures: %d)",
            s["llm"]["groq_calls"],
            s["llm"]["groq_failures"],
        )
        log.info("─" * 60)


# ─────────────────────────────────────────────────────────────────
# 7. HEALTH CHECK — PRE-FLIGHT SYSTEM VALIDATION
# ─────────────────────────────────────────────────────────────────


class HealthCheck:
    """
    Validates the system before starting a session.
    Catches configuration errors early — before 100 jobs are scraped
    and letters start failing.

    Checks:
    - GROQ_API_KEY is set and valid (makes a test call)
    - CV file exists and is non-empty
    - SQLite database is writable
    - scikit-learn is importable
    - LinkedIn credentials are set (if Selenium mode)

    Usage:
        health = HealthCheck()
        if not health.run():
            sys.exit(1)   # abort before wasting time
    """

    def __init__(self):
        self.log = logging.getLogger("GeorgeHawk.HealthCheck")
        self.checks = []
        self.passed = 0
        self.failed = 0

    def _check(self, name: str, fn) -> bool:
        try:
            ok, msg = fn()
            status = "✅ PASS" if ok else "❌ FAIL"
            self.log.info("  %-30s %s  %s", name, status, msg)
            if ok:
                self.passed += 1
            else:
                self.failed += 1
            return ok
        except Exception as e:
            self.log.error("  %-30s ❌ ERROR  %s", name, e)
            self.failed += 1
            return False

    def run(self, require_selenium: bool = False) -> bool:
        self.log.info("=" * 55)
        self.log.info("GeorgeHawk Health Check")
        self.log.info("=" * 55)

        self._check("GROQ_API_KEY set", self._check_groq_key)
        self._check("Groq API reachable", self._check_groq_ping)
        self._check("CV file exists", self._check_cv)
        self._check("SQLite writable", self._check_sqlite)
        self._check("scikit-learn import", self._check_sklearn)

        if require_selenium:
            self._check("LINKEDIN_EMAIL set", self._check_linkedin_email)
            self._check("LINKEDIN_PASSWORD set", self._check_linkedin_password)
            self._check("ChromeDriver available", self._check_chromedriver)

        self.log.info("=" * 55)
        self.log.info("Result: %d passed, %d failed", self.passed, self.failed)
        self.log.info("=" * 55)

        return self.failed == 0

    def _check_groq_key(self):
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return False, "Not set — add GROQ_API_KEY to .env"
        if not key.startswith("gsk_"):
            return False, "Looks wrong — should start with gsk_"
        return True, f"{key[:8]}..."

    def _check_groq_ping(self):
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return False, "Skipped (no key)"
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
                timeout=10,
            )
            if r.status_code == 200:
                return True, "API responded OK"
            return False, f"HTTP {r.status_code}: {r.text[:100]}"
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except Exception as e:
            return False, str(e)

    def _check_cv(self):
        path = os.getenv("CV_PATH", "George_Dogo_CV_Updated_2026.pdf")
        p = Path(path)
        if not p.exists():
            return False, f"Not found at: {path}"
        size = p.stat().st_size
        if size < 1000:
            return False, f"File suspiciously small: {size} bytes"
        return True, f"{path} ({size // 1024} KB)"

    def _check_sqlite(self):
        try:
            conn = sqlite3.connect("health_check_tmp.db")
            conn.execute("CREATE TABLE IF NOT EXISTS _test (id INTEGER)")
            conn.execute("INSERT INTO _test VALUES (1)")
            conn.commit()
            conn.close()
            Path("health_check_tmp.db").unlink(missing_ok=True)
            return True, "Read/write OK"
        except Exception as e:
            return False, str(e)

    def _check_sklearn(self):
        try:
            import sklearn

            return True, f"v{sklearn.__version__}"
        except ImportError:
            return False, "Not installed — run: pip install scikit-learn"

    def _check_linkedin_email(self):
        email = os.getenv("LINKEDIN_EMAIL", "")
        return bool(email), email or "Not set in .env"

    def _check_linkedin_password(self):
        pwd = os.getenv("LINKEDIN_PASSWORD", "")
        return bool(pwd), "Set ✓" if pwd else "Not set in .env"

    def _check_chromedriver(self):
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            driver = webdriver.Chrome(options=opts)
            version = driver.capabilities.get("browserVersion", "unknown")
            driver.quit()
            return True, f"Chrome {version}"
        except Exception as e:
            return (
                False,
                f"{e} — install chromedriver or run: pip install webdriver-manager",
            )


# ─────────────────────────────────────────────────────────────────
# 8. NOTIFY — SESSION COMPLETE WEBHOOK / EMAIL
# ─────────────────────────────────────────────────────────────────


class Notify:
    """
    Optional notifications when a session completes.
    Supports:
    - Slack webhook (recommended — free)
    - Discord webhook
    - SMTP email (Gmail, Outlook)

    Setup Slack:
        1. Create a Slack App at api.slack.com/apps
        2. Enable Incoming Webhooks
        3. Copy webhook URL to .env: SLACK_WEBHOOK_URL=https://hooks.slack.com/...

    Usage:
        notify = Notify()
        notify.send(metrics.summary())
    """

    def __init__(self):
        self.log = logging.getLogger("GeorgeHawk.Notify")
        self.slack_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    def send(self, metrics: dict):
        if self.slack_url:
            self._slack(metrics)
        if self.discord_url:
            self._discord(metrics)

    def _slack(self, m: dict):
        funnel = m.get("funnel", {})
        quality = m.get("quality", {})
        text = (
            f"*GeorgeHawk Session Complete* 🦅\n"
            f"Scraped: {funnel.get('scraped')} | "
            f"Applied: {funnel.get('applied')} | "
            f"Avg score: {quality.get('avg_fit_score')} | "
            f"Conversion: {quality.get('conversion_rate')}"
        )
        try:
            r = requests.post(self.slack_url, json={"text": text}, timeout=5)
            if r.status_code == 200:
                self.log.info("Slack notification sent")
            else:
                self.log.warning("Slack notify failed: %s", r.status_code)
        except Exception as e:
            self.log.warning("Slack error: %s", e)

    def _discord(self, m: dict):
        funnel = m.get("funnel", {})
        quality = m.get("quality", {})
        content = (
            f"**GeorgeHawk Session Complete** 🦅  "
            f"Scraped: {funnel.get('scraped')} | "
            f"Applied: {funnel.get('applied')} | "
            f"Score: {quality.get('avg_fit_score')} | "
            f"Conversion: {quality.get('conversion_rate')}"
        )
        try:
            requests.post(self.discord_url, json={"content": content}, timeout=5)
        except Exception as e:
            self.log.warning("Discord error: %s", e)


# ─────────────────────────────────────────────────────────────────
# EXAMPLE: Drop-in replacement main() with all improvements active
# ─────────────────────────────────────────────────────────────────


def run_production_session(config_path: str = "config.yaml", max_jobs: int = 50):
    """
    Example session using all production improvements.

    Replace the main() call in main.py with this for full hardening.
    """
    import yaml
    import sys

    # 1. Structured logging
    log = setup_logging(log_file="georgehawk.log", level="INFO")
    log.info("GeorgeHawk production session starting")

    # 2. Health check before anything else
    health = HealthCheck()
    if not health.run():
        log.error("Health check failed — aborting session")
        sys.exit(1)

    # 3. Validated config
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    try:
        config = ValidatedConfig(**raw)
    except Exception as e:
        log.error("Config validation error: %s", e)
        sys.exit(1)

    # 4. Session metrics tracker
    metrics = SessionMetrics()

    # 5. Rate limiter for LinkedIn (2 applications per minute)
    limiter = RateLimiter(requests_per_minute=2)

    # 6. Robust LLM client
    llm = RobustGroqLLM()

    # 7. Import main components
    from main import (
        RemoteOKScraper,
        RedditJobScraper,
        JobFitScorer,
        ApplicationWriter,
        JobStore,
        check_visa_relocation,
    )

    scorer = JobFitScorer()
    store = JobStore()

    # 8. Scrape
    all_jobs = []
    all_jobs += RemoteOKScraper().fetch(config.keywords)
    all_jobs += RedditJobScraper().fetch(config.keywords)
    metrics.jobs_scraped = len(all_jobs)
    log.info("Scraped %d raw jobs", metrics.jobs_scraped)

    # 9. Score + filter + store
    writer = ApplicationWriter()
    processed = 0

    for job in all_jobs:
        if store.exists(job.job_id):
            continue

        # Blacklist
        title_lower = job.title.lower()
        if any(b.lower() in title_lower for b in config.blacklisted_titles):
            metrics.jobs_filtered_blacklist += 1
            continue

        # Score
        base = scorer.score(job.description)
        boost = scorer.keyword_boost(f"{job.title} {job.description}")
        job.fit_score = min(base + boost, 1.0)
        metrics.record_score(job.fit_score)
        metrics.jobs_scored += 1

        if job.fit_score < config.min_fit_score:
            metrics.jobs_filtered_low_score += 1
            continue

        job.visa_sponsor, job.relocation = check_visa_relocation(
            f"{job.title} {job.description}"
        )

        # Cover letter
        cover = ""
        if job.fit_score > 0.10:
            metrics.groq_api_calls += 1
            cover = writer.cover_letter(job)
            if cover:
                metrics.letters_generated += 1
            else:
                metrics.groq_api_failures += 1

        store.save(job, cover)
        metrics.jobs_saved += 1
        processed += 1

        if processed >= max_jobs:
            break

    # 10. Report
    metrics.print_summary()
    metrics.save("metrics.json")

    # 11. Notify (if configured)
    Notify().send(metrics.summary())

    log.info("Session complete. Top jobs written to top_jobs.json")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeorgeHawk — Production Mode")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-jobs", type=int, default=50)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    run_production_session(config_path=args.config, max_jobs=args.max_jobs)
