import os
import json
import time
import hashlib
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException

from bs4 import BeautifulSoup
from google.cloud import storage

# =====================
# CONFIG
# =====================

START_URL = "https://www.midcindia.org"
ALLOWED_DOMAIN = "midcindia.org"
BUCKET_NAME = "midc-general-chatbot-bucket-web-data"
MAX_DEPTH = 5
PAGE_TIMEOUT = 20

visited = set()

# =====================
# DRIVER SETUP
# =====================

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

# =====================
# UTILS
# =====================

def is_internal(url):
    return ALLOWED_DOMAIN in urlparse(url).netloc

def url_id(url):
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def upload_json(path, data):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")

# =====================
# PAGE EXTRACTION
# =====================

def extract_page(driver, url):
    page_data = {
        "url": url,
        "title": "",
        "meta": {},
        "text": "",
        "links": [],
        "forms": [],
        "buttons": []
    }

    soup = BeautifulSoup(driver.page_source, "html.parser")

    page_data["title"] = soup.title.text.strip() if soup.title else ""

    for meta in soup.find_all("meta"):
        if meta.get("name") and meta.get("content"):
            page_data["meta"][meta["name"]] = meta["content"]

    page_data["text"] = " ".join(soup.stripped_strings)

    # Links
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        page_data["links"].append({
            "text": a.get_text(strip=True),
            "url": href,
            "internal": is_internal(href)
        })

    # Forms
    for form in soup.find_all("form"):
        fields = []
        for inp in form.find_all(["input", "select", "textarea"]):
            fields.append({
                "name": inp.get("name"),
                "type": inp.get("type", inp.name),
                "required": inp.has_attr("required")
            })

        page_data["forms"].append({
            "action": form.get("action"),
            "method": form.get("method", "GET"),
            "fields": fields
        })

    # Buttons
    for btn in soup.find_all("button"):
        page_data["buttons"].append(btn.get_text(strip=True))

    return page_data

# =====================
# CRAWLER
# =====================

def crawl(url, depth=0):
    if depth > MAX_DEPTH or url in visited:
        return

    visited.add(url)
    driver = None

    try:
        driver = create_driver()
        driver.set_page_load_timeout(PAGE_TIMEOUT)
        driver.get(url)
        time.sleep(2)

        page = extract_page(driver, url)
        pid = url_id(url)

        upload_json(f"pages/{pid}.json", page)

        for link in page["links"]:
            if link["internal"]:
                crawl(link["url"], depth + 1)
            else:
                upload_json(
                    f"external_links/{url_id(link['url'])}.json",
                    link
                )

    except Exception as e:
        upload_json(
            f"errors/{url_id(url)}.json",
            {"url": url, "error": str(e)}
        )

    finally:
        if driver:
            driver.quit()

# =====================
# ENTRYPOINT
# =====================

if __name__ == "__main__":
    crawl(START_URL)
