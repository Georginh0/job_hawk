"""
selenium_applier.py — LinkedIn Easy Apply Automation
Part of GeorgeHawk system.

Uses Selenium + Chrome to automate LinkedIn Easy Apply.
Handles multi-step forms, file uploads, and text questions via Groq LLM.

LEGAL NOTE: Use responsibly. Comply with LinkedIn ToS.
Rate-limit your applications (max 20–30/day recommended).
"""

import os, time, logging, random
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, ElementClickInterceptedException
)

log = logging.getLogger("GeorgeHawk.Selenium")

# ─────────────────────────────────────────────
# BROWSER SETUP
# ─────────────────────────────────────────────
def create_driver(headless: bool = False) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--window-size=1920,1080")
    # Stealth user agent
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    # Remove webdriver flag to avoid detection
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ─────────────────────────────────────────────
# LINKEDIN LOGIN
# ─────────────────────────────────────────────
def linkedin_login(driver: webdriver.Chrome, email: str, password: str):
    driver.get("https://www.linkedin.com/login")
    wait = WebDriverWait(driver, 15)

    wait.until(EC.presence_of_element_located((By.ID, "username"))).send_keys(email)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    # Wait for home page
    try:
        wait.until(EC.url_contains("feed"))
        log.info("LinkedIn login successful")
    except TimeoutException:
        log.warning("Login may need manual verification — check browser")


# ─────────────────────────────────────────────
# JOB SEARCH
# ─────────────────────────────────────────────
def search_linkedin_jobs(
    driver: webdriver.Chrome,
    keywords: str,
    location: str,
    easy_apply_only: bool = True,
    remote: bool = True,
    experience: list[str] = None
) -> list[str]:
    """Returns list of job URLs from LinkedIn search."""
    base = "https://www.linkedin.com/jobs/search/?"
    params = f"keywords={keywords.replace(' ', '+')}&location={location.replace(' ', '+')}"
    if easy_apply_only:
        params += "&f_LF=f_AL"  # Easy Apply filter
    if remote:
        params += "&f_WT=2"     # Remote filter
    params += "&f_TPR=r604800"  # Posted in last week

    driver.get(base + params)
    time.sleep(random.uniform(2, 4))

    job_urls = []
    wait = WebDriverWait(driver, 10)

    # Scroll and collect job cards
    for _ in range(5):  # scroll 5 pages
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "a.job-card-container__link")
            for card in cards:
                href = card.get_attribute("href")
                if href and "/jobs/view/" in href and href not in job_urls:
                    job_urls.append(href.split("?")[0])
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.5, 3))
        except Exception:
            break

    log.info(f"Found {len(job_urls)} LinkedIn job URLs")
    return job_urls[:50]  # cap at 50


# ─────────────────────────────────────────────
# EASY APPLY FORM HANDLER
# ─────────────────────────────────────────────
class EasyApplyHandler:
    """
    Handles the LinkedIn Easy Apply multi-step modal.
    Uses Groq LLM to answer text questions dynamically.
    """

    GEORGE_ANSWERS = {
        # Common LinkedIn form fields — pre-filled
        "first name": "George",
        "last name": "Dogo",
        "email": os.getenv("LINKEDIN_EMAIL", "George_dogo@aol.com"),
        "phone": os.getenv("PHONE", "+2347086276797"),
        "city": "Abuja",
        "country": "Nigeria",
        "linkedin": "https://www.linkedin.com/in/george-dogo",
        "github": "https://github.com/Georginh0",
        "website": "https://github.com/Georginh0",
        "years of experience": "4",
        "salary": "negotiable",
        "expected salary": "negotiable",
        "visa": "Yes",  # willing to relocate
        "sponsorship": "Yes",
        "authorized": "Yes — willing to obtain visa with sponsorship",
        "relocate": "Yes",
        "remote": "Yes",
        "notice period": "1 month",
        "start date": "1 month",
        "cover letter": "",  # filled dynamically
    }

    def __init__(self, driver: webdriver.Chrome, writer=None):
        self.driver = driver
        self.writer = writer  # ApplicationWriter instance
        self.wait = WebDriverWait(driver, 10)

    def _find_answer(self, label: str, job=None) -> str:
        label_lower = label.lower().strip()
        # Exact match
        for key, val in self.GEORGE_ANSWERS.items():
            if key in label_lower:
                return val
        # Dynamic LLM answer for unknown questions
        if self.writer and job:
            return self.writer.answer_question(label, job)
        return ""

    def handle_text_input(self, element, label: str, job=None):
        answer = self._find_answer(label, job)
        if answer:
            element.clear()
            element.send_keys(answer)

    def handle_select(self, element, label: str):
        select = Select(element)
        label_lower = label.lower()
        if "year" in label_lower:
            try:
                select.select_by_value("4")
            except Exception:
                select.select_by_index(4)
        elif "country" in label_lower or "nigeria" in label_lower:
            try:
                select.select_by_visible_text("Nigeria")
            except Exception:
                pass
        else:
            # Select first non-empty option
            for opt in select.options:
                if opt.text.strip() and opt.get_attribute("value"):
                    select.select_by_visible_text(opt.text)
                    break

    def handle_radio(self, elements, label: str):
        label_lower = label.lower()
        for el in elements:
            val = el.get_attribute("value") or ""
            sibling = el.find_element(By.XPATH, "following-sibling::label")
            txt = sibling.text.lower() if sibling else val.lower()

            # For yes/no questions about willing to relocate/work remotely
            if any(kw in label_lower for kw in ["relocate", "sponsor", "remote", "authorized", "willing"]):
                if "yes" in txt or "true" in val.lower():
                    el.click()
                    return
            # Default: first option
            el.click()
            return

    def upload_cv(self, job=None):
        cv_path = os.getenv("CV_PATH", "George_Dogo_CV_Updated_2026.pdf")
        if not Path(cv_path).exists():
            log.warning(f"CV not found at {cv_path}")
            return
        try:
            upload = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            upload.send_keys(str(Path(cv_path).absolute()))
            log.info("CV uploaded")
            time.sleep(1)
        except NoSuchElementException:
            pass

    def fill_modal(self, job=None) -> bool:
        """Fill one page of the Easy Apply modal. Returns True if done."""
        try:
            modal = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".jobs-easy-apply-modal"))
            )
        except TimeoutException:
            return True  # Modal not found — already closed or applied

        # Handle file upload
        self.upload_cv(job)

        # Find all form groups
        form_groups = modal.find_elements(By.CSS_SELECTOR, ".jobs-easy-apply-form-section")
        for group in form_groups:
            try:
                # Get label
                try:
                    label = group.find_element(By.CSS_SELECTOR, "label, legend").text
                except Exception:
                    label = ""

                # Text inputs
                inputs = group.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='email'], input[type='tel'], textarea")
                for inp in inputs:
                    self.handle_text_input(inp, label, job)

                # Number inputs
                num_inputs = group.find_elements(By.CSS_SELECTOR, "input[type='number']")
                for inp in num_inputs:
                    if "year" in label.lower():
                        inp.clear(); inp.send_keys("4")
                    elif "salary" in label.lower():
                        inp.clear(); inp.send_keys("0")  # prefer not to disclose

                # Selects
                selects = group.find_elements(By.TAG_NAME, "select")
                for sel in selects:
                    self.handle_select(sel, label)

                # Radio buttons
                radios = group.find_elements(By.CSS_SELECTOR, "input[type='radio']")
                if radios:
                    self.handle_radio(radios, label)

            except Exception as e:
                log.debug(f"Form group error: {e}")

        # Try to proceed (Next / Review / Submit)
        try:
            # Look for Next button first
            next_btn = modal.find_element(
                By.CSS_SELECTOR,
                "button[aria-label='Continue to next step'], button[aria-label='Review your application']"
            )
            next_btn.click()
            time.sleep(random.uniform(1, 2))
            return False  # More steps remain
        except NoSuchElementException:
            pass

        # Submit button
        try:
            submit = modal.find_element(By.CSS_SELECTOR, "button[aria-label='Submit application']")
            submit.click()
            log.info(f"✅ Application submitted: {job.title if job else 'unknown'}")
            time.sleep(random.uniform(2, 4))
            return True
        except NoSuchElementException:
            pass

        return False


# ─────────────────────────────────────────────
# FULL APPLY FLOW
# ─────────────────────────────────────────────
def apply_to_job(driver: webdriver.Chrome, url: str, job=None, writer=None) -> bool:
    """Navigate to job page and attempt Easy Apply."""
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 4))
        wait = WebDriverWait(driver, 10)

        # Click Easy Apply button
        try:
            btn = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button.jobs-apply-button, .jobs-apply-button--top-card")
                )
            )
            if "easy apply" not in btn.text.lower():
                log.info(f"No Easy Apply for: {url}")
                return False
            btn.click()
            time.sleep(1)
        except TimeoutException:
            log.debug(f"Easy Apply button not found: {url}")
            return False

        # Handle multi-step form (max 10 steps)
        handler = EasyApplyHandler(driver, writer)
        for step in range(10):
            done = handler.fill_modal(job)
            if done:
                return True
            time.sleep(random.uniform(1, 2))

        return False

    except Exception as e:
        log.error(f"Apply error for {url}: {e}")
        return False


# ─────────────────────────────────────────────
# MAIN LINKEDIN SESSION
# ─────────────────────────────────────────────
def run_linkedin_session(
    email: str,
    password: str,
    keywords: list[str],
    locations: list[str],
    max_applications: int = 20,
    writer=None,
    store=None
):
    """Full LinkedIn job hunt session."""
    driver = create_driver(headless=False)  # headless=True once confident
    applied = 0

    try:
        linkedin_login(driver, email, password)
        time.sleep(random.uniform(3, 5))

        for location in locations:
            for kw in keywords[:2]:  # 2 keywords per location
                if applied >= max_applications:
                    break

                urls = search_linkedin_jobs(driver, kw, location)
                for url in urls:
                    if applied >= max_applications:
                        break

                    # Check not already applied
                    if store and store.exists(url[:50]):
                        continue

                    success = apply_to_job(driver, url, writer=writer)
                    if success:
                        applied += 1
                        log.info(f"Applied {applied}/{max_applications}: {url}")
                        if store:
                            store.mark_applied(url[:12])
                        # Human-like pause between applications
                        time.sleep(random.uniform(10, 20))

    finally:
        driver.quit()
        log.info(f"LinkedIn session complete. Applied: {applied}")
    return applied
