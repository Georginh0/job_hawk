"""
scrapers.py — GeorgeHawk Job Scrapers
======================================
All job source scrapers in one module. Drop into project root.

Sources implemented:
  ✅ RemoteOK    — free public JSON API (fixed: filter on tags, not just title)
  ✅ Arbeitnow   — free public JSON API, 1,000+ EU/remote jobs daily
  ✅ Remotive    — free public JSON API, curated remote roles
  ✅ Himalayas   — free public JSON API, startup/tech focused
  ✅ WWR RSS     — We Work Remotely RSS feed, parsed with feedparser
  ✅ Reddit      — r/forhire + r/remotework (improved filtering)
  ⚠️ Wellfound   — Selenium-based, see WellfoundScraper below
  ⚠️ LinkedIn    — Selenium-based, lives in selenium_applier.py

Usage:
    from scrapers import get_all_jobs
    jobs = get_all_jobs(keywords=["data scientist", "ml engineer"])
"""

import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import requests

log = logging.getLogger("GeorgeHawk.Scrapers")

HEADERS = {"User-Agent": "GeorgeHawk/1.0 (+https://github.com/Georginh0/georgehawk)"}


# ─────────────────────────────────────────────
# JOB DATACLASS (shared)
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
        if not self.job_id:
            self.job_id = hashlib.md5(
                f"{self.title}{self.company}{self.url}".encode()
            ).hexdigest()[:12]


# ─────────────────────────────────────────────
# SHARED HELPER
# ─────────────────────────────────────────────
def _keyword_match(text: str, keywords: list[str]) -> bool:
    """
    FIX for BUG 1: match against full text (title + description + tags),
    not just title. Previous code only checked job title — too strict.
    """
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _safe_get(
    url: str, params: dict = None, timeout: int = 15
) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.Timeout:
        log.warning("Timeout fetching %s", url)
    except requests.exceptions.HTTPError as e:
        log.warning("HTTP %s from %s", e.response.status_code, url)
    except requests.exceptions.RequestException as e:
        log.error("Request error for %s: %s", url, e)
    return None


# ─────────────────────────────────────────────
# 1. REMOTEOK — FIXED
# ─────────────────────────────────────────────
class RemoteOKScraper:
    """
    RemoteOK public JSON API.

    BUG 1 FIX: The original code only matched keywords against job title.
    RemoteOK titles like "Senior Engineer" don't contain "data scientist"
    so everything got filtered. Now matches against title + description + tags.
    """

    URL = "https://remoteok.com/api"

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []
        r = _safe_get(self.URL)
        if not r:
            return jobs

        try:
            data = r.json()
        except Exception:
            log.error("RemoteOK: invalid JSON response")
            return jobs

        for item in data[1:]:  # first element is legal notice
            title = item.get("position", "") or ""
            description = item.get("description", "") or ""
            tags = " ".join(item.get("tags", []) or [])

            # FIXED: match against title + description + tags
            search_text = f"{title} {description} {tags}"
            if not _keyword_match(search_text, keywords):
                continue

            jobs.append(
                Job(
                    title=title,
                    company=item.get("company", ""),
                    location=item.get("location", "Remote") or "Remote",
                    url=item.get("url", ""),
                    description=description[:2000],
                    source="RemoteOK",
                    salary=item.get("salary", ""),
                    remote=True,
                )
            )

        log.info("RemoteOK: %d jobs matched (from %d total)", len(jobs), len(data) - 1)
        return jobs


# ─────────────────────────────────────────────
# 2. ARBEITNOW — NEW
# ─────────────────────────────────────────────
class ArbeitnowScraper:
    """
    Arbeitnow free public API.
    Returns 100+ EU-focused and remote jobs. No auth required.
    Great for UK, Germany, Netherlands, Switzerland roles.
    API docs: https://arbeitnow.com/api/job-board-api
    """

    URL = "https://www.arbeitnow.com/api/job-board-api"

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []
        page = 1
        max_pages = 5  # 100 jobs per page, 5 pages = 500 candidates

        while page <= max_pages:
            r = _safe_get(self.URL, params={"page": page})
            if not r:
                break

            try:
                data = r.json()
            except Exception:
                break

            items = data.get("data", [])
            if not items:
                break

            for item in items:
                title = item.get("title", "") or ""
                description = item.get("description", "") or ""
                tags = " ".join(item.get("tags", []) or [])

                if not _keyword_match(f"{title} {description} {tags}", keywords):
                    continue

                jobs.append(
                    Job(
                        title=title,
                        company=item.get("company_name", ""),
                        location=item.get("location", "Remote"),
                        url=item.get("url", ""),
                        description=description[:2000],
                        source="Arbeitnow",
                        remote=item.get("remote", False),
                        visa_sponsor="visa" in description.lower()
                        or "sponsor" in description.lower(),
                    )
                )

            page += 1
            time.sleep(0.5)  # polite rate limiting

        log.info("Arbeitnow: %d jobs matched", len(jobs))
        return jobs


# ─────────────────────────────────────────────
# 3. REMOTIVE — NEW
# ─────────────────────────────────────────────
class RemotiveScraper:
    """
    Remotive free public API.
    Curated remote tech jobs. Categories: software-dev, data, devops.
    API docs: https://remotive.com/api
    """

    URL = "https://remotive.com/api/remote-jobs"
    CATEGORIES = ["software-dev", "data", "devops-sysadmin"]

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []

        for category in self.CATEGORIES:
            r = _safe_get(self.URL, params={"category": category, "limit": 100})
            if not r:
                continue

            try:
                data = r.json()
            except Exception:
                continue

            for item in data.get("jobs", []):
                title = item.get("title", "") or ""
                description = item.get("description", "") or ""
                tags = " ".join(item.get("tags", []) or [])

                if not _keyword_match(f"{title} {description} {tags}", keywords):
                    continue

                jobs.append(
                    Job(
                        title=title,
                        company=item.get("company_name", ""),
                        location=item.get("candidate_required_location", "Remote"),
                        url=item.get("url", ""),
                        description=description[:2000],
                        source="Remotive",
                        salary=item.get("salary", ""),
                        remote=True,
                    )
                )

            time.sleep(0.5)

        log.info("Remotive: %d jobs matched", len(jobs))
        return jobs


# ─────────────────────────────────────────────
# 4. HIMALAYAS — NEW
# ─────────────────────────────────────────────
class HimalayasScraper:
    """
    Himalayas free public API.
    Startup and scale-up remote roles, salary-transparent.
    API docs: https://himalayas.app/jobs/api
    """

    URL = "https://himalayas.app/jobs/api"

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []
        r = _safe_get(self.URL, params={"limit": 100})
        if not r:
            return jobs

        try:
            data = r.json()
        except Exception:
            log.error("Himalayas: invalid JSON")
            return jobs

        for item in data.get("jobs", []):
            title = item.get("title", "") or ""
            description = item.get("description", "") or ""
            skills = " ".join(item.get("requiredSkills", []) or [])

            if not _keyword_match(f"{title} {description} {skills}", keywords):
                continue

            jobs.append(
                Job(
                    title=title,
                    company=item.get("companyName", ""),
                    location=item.get("locationRestrictions", ["Remote"])[0]
                    if item.get("locationRestrictions")
                    else "Remote",
                    url=f"https://himalayas.app/jobs/{item.get('slug', '')}",
                    description=description[:2000],
                    source="Himalayas",
                    salary=str(item.get("salaryCurrency", ""))
                    + " "
                    + str(item.get("salaryMin", "")),
                    remote=True,
                )
            )

        log.info("Himalayas: %d jobs matched", len(jobs))
        return jobs


# ─────────────────────────────────────────────
# 5. WE WORK REMOTELY — RSS FEED
# ─────────────────────────────────────────────
class WWRScraper:
    """
    We Work Remotely RSS feeds.
    High-quality curated remote jobs. No auth needed.
    """

    FEEDS = [
        "https://weworkremotely.com/categories/remote-data-science-jobs.rss",
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-machine-learning-ai-jobs.rss",
    ]

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []

        try:
            import feedparser
        except ImportError:
            log.warning(
                "WWR scraper requires feedparser. Install with: pip install feedparser"
            )
            return jobs

        for feed_url in self.FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "") or ""
                    summary = entry.get("summary", "") or ""

                    if not _keyword_match(f"{title} {summary}", keywords):
                        continue

                    # Parse company from title (format: "Company: Role")
                    company = ""
                    if ": " in title:
                        company, title = title.split(": ", 1)

                    jobs.append(
                        Job(
                            title=title.strip(),
                            company=company.strip(),
                            location="Remote",
                            url=entry.get("link", ""),
                            description=summary[:2000],
                            source="WeWorkRemotely",
                            remote=True,
                        )
                    )

                time.sleep(0.5)
            except Exception as e:
                log.warning("WWR feed error (%s): %s", feed_url, e)

        log.info("We Work Remotely: %d jobs matched", len(jobs))
        return jobs


# ─────────────────────────────────────────────
# 6. REDDIT — IMPROVED
# ─────────────────────────────────────────────
class RedditJobScraper:
    """
    Reddit job posts — improved filtering.
    Targets actual hiring posts, not meta discussion.
    """

    SUBREDDITS = ["forhire", "remotework", "MachineLearning", "datascience"]

    # Hiring signal phrases — expanded from original
    HIRING_SIGNALS = [
        "[hiring]",
        "[h]",
        "we are hiring",
        "we're hiring",
        "looking for a",
        "looking to hire",
        "seeking a",
        "job opportunity",
        "open position",
        "job opening",
        "now hiring",
    ]

    def fetch(self, keywords: list[str]) -> list[Job]:
        jobs = []
        base = "https://www.reddit.com/r/{sub}/search.json"

        for sub in self.SUBREDDITS:
            for kw in keywords[:3]:
                try:
                    r = _safe_get(
                        base.format(sub=sub),
                        params={"q": kw, "sort": "new", "limit": 25, "restrict_sr": 1},
                    )
                    if not r:
                        continue

                    for post in r.json()["data"]["children"]:
                        d = post["data"]
                        title = d.get("title", "") or ""
                        body = d.get("selftext", "") or ""

                        # IMPROVED: check hiring signals in title (case-insensitive)
                        title_lower = title.lower()
                        is_hiring = any(
                            sig in title_lower for sig in self.HIRING_SIGNALS
                        )
                        if not is_hiring:
                            continue

                        # Also require keyword match somewhere in post
                        if not _keyword_match(f"{title} {body}", keywords):
                            continue

                        jobs.append(
                            Job(
                                title=title[:100],
                                company="via Reddit",
                                location="Remote",
                                url=f"https://reddit.com{d.get('permalink', '')}",
                                description=body[:2000],
                                source=f"Reddit/r/{sub}",
                                remote=True,
                            )
                        )

                    time.sleep(1.5)  # Reddit rate limit

                except Exception as e:
                    log.debug("Reddit r/%s error: %s", sub, e)

        log.info("Reddit: %d hiring posts matched", len(jobs))
        return jobs


# ─────────────────────────────────────────────
# 7. WELLFOUND (SKELETON — SELENIUM REQUIRED)
# ─────────────────────────────────────────────
class WellfoundScraper:
    """
    Wellfound (AngelList Talent) — startup jobs.

    Requires Selenium because Wellfound is JavaScript-rendered.
    This skeleton shows what to implement — connect to selenium_applier.py
    for the full browser session.

    Quick start without Selenium:
    - Go to https://wellfound.com/jobs?remote=true&role=data-scientist
    - Export the page (Ctrl+S) and parse with BeautifulSoup
    - Or use the official Talent API if you have recruiter access

    To activate: call from selenium_applier.py after LinkedIn login,
    using the same Chrome session (already authenticated).
    """

    def fetch(self, keywords: list[str], driver=None) -> list[Job]:
        if driver is None:
            log.info(
                "Wellfound: Selenium driver required. "
                "Pass an active Chrome driver to enable scraping. "
                "See selenium_applier.py for integration."
            )
            return []

        # Placeholder — implement with driver.get() + BeautifulSoup parsing
        # URL format: https://wellfound.com/jobs?role=data-scientist&remote=true
        jobs = []
        log.info("Wellfound: Selenium scraping not yet implemented")
        return jobs


# ─────────────────────────────────────────────
# AGGREGATE — SINGLE ENTRY POINT
# ─────────────────────────────────────────────
def get_all_jobs(keywords: list[str], include_reddit: bool = True) -> list[Job]:
    """
    Run all available scrapers and return deduplicated job list.
    Called from production.py instead of importing scrapers individually.

    Args:
        keywords: list of search keywords from config.yaml
        include_reddit: Reddit posts are lower quality — set False for pure job boards

    Returns:
        Deduplicated list of Job objects, all sources combined.
    """
    all_jobs: list[Job] = []
    seen_ids: set[str] = set()

    scrapers = [
        ("RemoteOK", RemoteOKScraper().fetch, {"keywords": keywords}),
        ("Arbeitnow", ArbeitnowScraper().fetch, {"keywords": keywords}),
        ("Remotive", RemotiveScraper().fetch, {"keywords": keywords}),
        ("Himalayas", HimalayasScraper().fetch, {"keywords": keywords}),
        ("WWR", WWRScraper().fetch, {"keywords": keywords}),
    ]

    if include_reddit:
        scrapers.append(("Reddit", RedditJobScraper().fetch, {"keywords": keywords}))

    for name, fn, kwargs in scrapers:
        try:
            results = fn(**kwargs)
            # Deduplicate across sources
            before = len(all_jobs)
            for job in results:
                if job.job_id not in seen_ids:
                    seen_ids.add(job.job_id)
                    all_jobs.append(job)
            new = len(all_jobs) - before
            log.info("%-15s → %2d new jobs (deduped)", name, new)
        except Exception as e:
            log.error("Scraper %s failed: %s", name, e)

    log.info("Total unique jobs collected: %d", len(all_jobs))
    return all_jobs
