"""
Tourism Website Crawler (Selenium + BeautifulSoup)

Purpose:
- Crawl tourism bureau websites (same-domain only)
- Extract clean text content from pages
- Filter out low-value/junk links (login, search, social, downloads, etc.)
- Rank discovered links by tourism relevance
- Return (text, metadata) pairs for downstream chunking + vector indexing

Design choices:
- Selenium is used to render JavaScript-heavy pages.
- BeautifulSoup is used for parsing and cleaning HTML.
- BFS crawling is used to explore site pages progressively.
- A character threshold is used to skip thin-content pages.
"""

from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from collections import deque
import time
import re
import os
import platform
import shutil
import requests

try:
    # Optional dependency for reliable driver installation on local/dev machines.
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:
    ChromeDriverManager = None


# ==================== DRIVER SETUP ====================
def create_driver():
    """
    Build a headless Chromium driver tuned for Linux server crawling.
    """
    options = Options()
    options.page_load_strategy = "eager"

    # Only force binary path when running on Linux and Chromium exists there.
    # On Windows/macOS, let Selenium discover installed Chrome automatically.
    if platform.system().lower() == "linux":
        for candidate in ("/usr/bin/chromium-browser", "/usr/bin/chromium", "/usr/bin/google-chrome"):
            if os.path.exists(candidate):
                options.binary_location = candidate
                break

    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    )
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--host-rules=MAP analytics.google.com 127.0.0.1,"
                         "MAP googletagmanager.com 127.0.0.1,"
                         "MAP connect.facebook.net 127.0.0.1,"
                         "MAP hotjar.com 127.0.0.1")

    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    def _is_valid_driver_binary(path: str) -> bool:
        """Basic sanity checks to avoid launching non-executable notice files."""
        if not path or not os.path.isfile(path):
            return False
        filename = os.path.basename(path).lower()
        if "third_party_notices" in filename or not filename.startswith("chromedriver"):
            return False
        return os.access(path, os.X_OK)

    try:
        driver_path = None

        # 1) explicit override
        env_driver = os.getenv("CHROMEDRIVER_PATH")
        if _is_valid_driver_binary(env_driver):
            driver_path = env_driver

        # 2) system-installed chromedriver (Ubuntu apt, etc.)
        if not driver_path:
            for candidate in ("chromedriver", "chromium-driver"):
                resolved = shutil.which(candidate)
                if _is_valid_driver_binary(resolved):
                    driver_path = resolved
                    break

        # 3) webdriver-manager fallback
        if not driver_path and ChromeDriverManager:
            installed = ChromeDriverManager().install()
            if installed and os.path.isfile(installed):
                if _is_valid_driver_binary(installed):
                    driver_path = installed
                else:
                    driver_dir = os.path.dirname(installed)
                    for candidate in ("chromedriver.exe", "chromedriver"):
                        alt_path = os.path.join(driver_dir, candidate)
                        if _is_valid_driver_binary(alt_path):
                            driver_path = alt_path
                            break
            if not driver_path:
                raise RuntimeError(
                    f"webdriver-manager returned non-executable driver path: {installed}"
                )

        if driver_path:
            service = Service(driver_path)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            # Last attempt: let Selenium manager resolve driver automatically.
            driver = webdriver.Chrome(options=options)

        driver.set_page_load_timeout(int(os.getenv("CRAWL_PAGELOAD_TIMEOUT", "60")))
        return driver
    except Exception as e:
        print(f"❌ Failed to initialize Chrome driver: {e}")
        print("💡 Install Google Chrome/Chromium and ensure ChromeDriver is executable.")
        print("💡 Ubuntu example: sudo apt install chromium-browser chromium-chromedriver")
        print("💡 Optional override: export CHROMEDRIVER_PATH=/usr/bin/chromedriver")
        print("💡 Optional: pip install webdriver-manager")
        return None


# Mapping domains to bureau names
BUREAU_MAP = {
    "visitethiopia.et": "Ethiopian Tourism Organization (National)",
    "visitoromia.org": "Oromia Tourism Commission",
    "visitamhara.travel": "Amhara Tourism Bureau",
    "tourismtigrai.com": "Tigray Tourism Bureau",
    "visitsidama.travel": "Sidama Tourism Bureau",
    "visitsouthethiopia.et": "South Ethiopia Tourism Bureau",
}

# The primary site that gets special crawl treatment (no link cap, no support filter)
PRIMARY_CRAWL_URL = "https://visitethiopia.et/"

# ==================== URL NORMALIZATION ====================
def normalize_url(url: str) -> str:
    """Normalize URL to reduce duplicate crawl targets."""
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower().replace("www.", "")
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")

    # Remove noisy tracking params while preserving meaningful ones.
    keep_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        k = key.lower()
        if k.startswith("utm_") or k in {"fbclid", "gclid", "ref", "source"}:
            continue
        keep_query.append((key, value))
    query = urlencode(sorted(keep_query))

    return urlunparse((scheme, netloc, path, "", query, ""))

# ==================== LINK EXTRACTION ====================
def get_all_links(base_url: str, soup: BeautifulSoup , start_url: str = ""):
    """
    Extract and filter internal links from one HTML page.
    Pipeline:
    1) Read href-like attributes from anchor tags
    2) Convert relative links -> absolute links
    3) Keep same-domain links only
    4) Apply junk filters (auth/admin/legal/search/download/etc.)
    5) Keep high-signal tourism links, skip noisy/generated URLs
    Returns:
    - set[str]: valid internal links for further crawling
    """
    links = set()

    auth_patterns = [
        "login", "signin", "log-in", "sign-in",
        "logout", "signout", "log-out", "sign-out",
        "register", "signup", "sign-up", "join",
        "account", "my-account", "user", "profile",
        "password", "reset", "forgot", "recover",
        "verify", "activation", "confirm",
    ]

    language_paths = ["/am/", "/ar/"]

    commerce_patterns = [
        "cart", "shopping-cart", "basket",
        "checkout", "payment", "pay",
        "invoice", "billing", "subscription", "store",
        "wishlist", "coupon",
    ]

    tech_patterns = [
        "api", "rest", "graphql", "json", "xml",
        "admin", "administrator", "dashboard",
        "control-panel", "cpanel", "backend",
        "config", "configuration", "setup",
        "install", "update", "upgrade", "migrate",
        "maintenance", "debug", "test", "staging",
        "dev",
    ]

    social_patterns = [
        "facebook", "twitter", "instagram", "linkedin",
        "youtube", "tiktok", "pinterest", "snapchat",
        "whatsapp", "telegram", "discord", "slack",
        "reddit", "tumblr", "flickr", "vk", "weibo",
    ]

    legal_patterns = [
        "terms", "conditions", "policy",
        "cookie", "disclaimer", "legal", "compliance",
        "gdpr", "ccpa", "copyright",
        "trademark", "license", "agreement",
    ]

    support_patterns = [
        "support","feedback", "report",
        "complaint",
        "chat", "live-chat", "messenger", "inbox",
    ]

    search_patterns = [
        "search", "find", "filter", "advanced-search",
        "results", "query", "q=", "?q=", "&q=",
        "sort=", "filter=", "category=", "tag=",
    ]



    file_patterns = [
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".ico",
        ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".flv",
        ".zip", ".rar", ".7z", ".tar", ".gz", ".tgz",
        ".xls", ".xlsx", ".ppt", ".pptx",
        ".exe", ".msi", ".dmg", ".apk", ".app",
        ".css", ".js", ".xml", ".rss", ".atom",
        ".csv", ".tsv", ".txt", ".log",
    ]

    archive_patterns = [
        "/archive/", "/archives/", "/202",
    ]

    interactive_patterns = [
        "rating", "vote",
        "like", "share", "follow", "subscribe",
        "newsletter", "mailing-list",
        "upload", "download", "print", "preview", "embed", "iframe",
    ]

    protocol_patterns = [
        "javascript:", "mailto:", "tel:", "sms:",
        "whatsapp:", "skype:", "weixin:", "tg:",
        "data:", "blob:", "ftp:",
    ]
    is_primary = normalize_url(start_url) == normalize_url(PRIMARY_CRAWL_URL)

    all_skip_patterns = (
        auth_patterns + commerce_patterns + tech_patterns + social_patterns +
        legal_patterns + search_patterns +
        interactive_patterns + archive_patterns + language_paths  +
         ([] if is_primary else support_patterns)
    )

    session_patterns = ["sessionid", "jsessionid", "sid=", "sess="]
    admin_paths = ["/wp-admin", "/administrator", "/admin/", "/manage/", "/backend"]
    content_filters = [ "?type=", "&type=","?format=", "&format=","?content=", "&content=", "?output=", "&output=", "?view=", "&view=","?print=", "&print=",]

    base_domain = urlparse(base_url).netloc

    # Use CSS selector correctly
    anchor_tags = soup.select("a[href], a[data-href], a[to], a[ng-href], a[v-bind\\:href]")

    for a_tag in anchor_tags:
        try:
            href = (
                a_tag.get("href")
                or a_tag.get("data-href")
                or a_tag.get("to")
                or a_tag.get("ng-href")
                or a_tag.get("v-bind:href")
            )

            if not href:
                continue

            href_lower = href.lower().strip()

            if any(href_lower.startswith(protocol) for protocol in protocol_patterns):
                continue

            if href.startswith("#") or href == "#":
                continue

            full_url = normalize_url(urljoin(base_url, href))

            # Remove fragment
            if "#" in full_url:
                full_url = full_url.split("#")[0]

            full_domain = urlparse(full_url).netloc
            if full_domain != base_domain:
                continue

            url_lower = full_url.lower()

            # Skip known junk patterns
            if any(pattern in url_lower for pattern in all_skip_patterns):
                continue

            # Skip file links
            if any(url_lower.endswith(ext) for ext in file_patterns):
                continue

            # Skip malformed/generated URLs
            special_char_count = sum(1 for c in full_url if c in "{}[]\\|;:")
            if special_char_count > 5:
                continue


            if any(pattern in url_lower for pattern in session_patterns):
                continue

            # Skip extreme pagination
            # if "page=" in url_lower or "p=" in url_lower:
            #     page_match = re.search(r"[?&](?:page|p)=(\d+)", url_lower)
            #     if page_match and int(page_match.group(1)) > 10:
            #         continue

            if any(path in url_lower for path in admin_paths):
                continue

            if any(filter_str in url_lower for filter_str in content_filters):
                continue

            links.add(full_url)

        except Exception:
            continue

    print(f"🔗 Found {len(links)} valid internal links")
    if links:
        sample_size = min(3, len(links))
        print(f"   Sample: {', '.join(list(links)[:sample_size])}")

    return links


def score_url_quality(url: str, base_url: str):
    """
      Heuristic scoring for crawl prioritization.
    Higher score => likely tourism-rich page.
    Lower score => likely noisy/auxiliary page.
    Signals:
    - +10 for high-value tourism patterns
    - +5 for medium-value patterns
    - -5 for low-value/news/gallery/resource patterns
    - short clean URLs and low parameter count get extra preference

    """
    score = 0
    url_lower = url.lower()

    high_value = [
        "/attraction", "/unesco", "/packages", "/package", "/destination", "/place", "/site",
        "/landmark", "/monument", "/heritage", "/rock-hewn", "/church", "/obelisk", "/stelae",
        "/culture", "/history", "/tradition", "/historical-sites", "/world-heritage", "/museum",
    ]

    medium_value = [
        "/tour", "/travel", "/visit", "/experience",
        "/festival", "/event", "/celebration",
        "/park", "/mountain", "/lake", "/river", "/endemic", "/wildlife", "/sanctuary",
        "/coffee-ceremony", "/cuisine", "/handicraft",
        "/meskel", "/timkat", "/ireecha", "/enqutatash",
        "/trekking", "/hiking", "/safari",
        "/bureau", "/office", "/regional",
    ]

    low_value = [
        "/news", "/blog", "/article", "/post",
        "/gallery", "/photo", "/image", "/video",
        "/download", "/resource",
    ]

    for pattern in high_value:
        if pattern in url_lower:
            score += 10

    for pattern in medium_value:
        if pattern in url_lower:
            score += 5

    for pattern in low_value:
        if pattern in url_lower:
            score -= 5

    # Slight preference for shorter URLs
    if len(url) - len(base_url) < 60:
        score += 3

    param_count = url.count("?") + url.count("&")
    if param_count == 0:
        score += 5
    elif param_count > 1:
        score -= param_count * 2

    return score


def get_all_links_sorted(base_url: str, soup: BeautifulSoup, start_url: str = ""):
    """
        Return candidate links sorted by descending quality score.
        This improves crawl efficiency by visiting high-value pages earlier.
    """
    links = get_all_links(base_url, soup, start_url=start_url)
    link_list = list(links)
    link_list.sort(key=lambda u: score_url_quality(u, base_url), reverse=True)
    return link_list


# ==================== TEXT EXTRACTION ====================
def extract_text(driver, url: str):
    """
     Render page with Selenium, then extract cleaned textual content.
    Steps:
    - Load page and wait for meaningful body text
    - Trigger lazy-loaded content via scroll down/up
    - Remove non-content tags (script/style/nav/footer/header/etc.)
    - Normalize whitespace
    Returns:
    - (text, soup) on success
    - ("", None) on failure
    """
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"🔍 Processing: {url} (attempt {attempt}/{max_attempts})")
            try:
                driver.get(url)
            except TimeoutException:
                # Some sites never fully finish loading on servers.
                # Keep parsing whatever DOM is already available.
                print("⚠️ Page load timeout, attempting to parse partial DOM...")

            wait = WebDriverWait(driver, 30)
            try:
                wait.until(lambda d: len(d.find_element(By.TAG_NAME, "body").text) > 50)
            except Exception:
                print("⚠️ Timeout waiting for text hydration, attempting extraction anyway...")
                time.sleep(3)

            # trigger lazy load
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
                tag.decompose()

            text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)

            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = text.strip()

            print(f"📄 Extracted length: {len(text)}")
            return text, soup
        except Exception as e:
            print(f"❌ Error extracting {url} (attempt {attempt}/{max_attempts}): {e}")
            if attempt == max_attempts:
                return "", None
            time.sleep(2)
    return "", None

def extract_text_via_http(url: str):
    """
    Fallback extraction using requests when Selenium renderer repeatedly times out.
    """
    try:
        print(f"🌐 HTTP fallback for: {url}")
        timeout = int(os.getenv("CRAWL_HTTP_TIMEOUT", "40"))
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
            tag.decompose()

        text = soup.body.get_text(separator="\n", strip=True) if soup.body else soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()

        print(f"📄 HTTP fallback extracted length: {len(text)}")
        return text, soup
    except Exception as e:
        print(f"❌ HTTP fallback failed for {url}: {e}")
        return "", None


# ==================== DATA EXTRACTION ====================
def get_data_from_website(driver, url: str):
    """
    Wrap extraction with metadata generation used by downstream RAG.
    Metadata fields:
    - url: source page URL
    - title: page title (fallback to h1)
    - bureau: bureau label inferred from domain
    Returns:
    - (text, metadata, soup)
    """
    try:
        text, soup = extract_text(driver, url)
        fallback_min_chars = int(os.getenv("CRAWL_HTTP_FALLBACK_MIN_CHARS", "30"))

        if soup is None:
            text, soup = extract_text_via_http(url)
            if soup is None:
                return "", {"url": url, "title": "Unknown"}, None
        elif len(text.strip()) < fallback_min_chars:
            # Selenium returned very thin content (often splash/blocked pages).
            # Try HTTP extraction and keep whichever has richer text.
            http_text, http_soup = extract_text_via_http(url)
            if http_soup is not None and len(http_text.strip()) > len(text.strip()):
                text, soup = http_text, http_soup

        title = "Visit Ethiopia"
        try:
            if soup.title and soup.title.get_text():
                title = soup.title.get_text().strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text().strip()
        except Exception:
            pass

        title = re.sub(r"\s+", " ", title)[:150]

        domain = urlparse(url).netloc.replace("www.", "")
        bureau_name = BUREAU_MAP.get(domain, "Visit Ethiopia Partner")

        metadata = {
            "url": url,
            "title": title,
            "bureau": bureau_name,
        }

        return text, metadata, soup

    except Exception as e:
        print(f"❌ Error processing {url}: {e}")
        return "", {"url": url, "title": "Unknown"}, None

# ==================== CRAWLER ====================
def crawl_website(start_url: str):
    """
    Crawl website using Breadth-First Search (BFS).
    BFS behavior:
    - Start from seed URL
    - Visit page, extract text, enqueue new links
    - Stop when queue is empty (or when max_pages is reached if provided)
    Storage rule:
    - Keep only pages with >= CRAWL_MIN_CONTENT_CHARS
      (default 120, avoids thin/empty pages in the vector store)
    Returns:
    - list[tuple[str, dict]] where each item is (clean_text, metadata)

    """
    driver = create_driver()
    if not driver:
        return []

    visited = set()
    queued = set([normalize_url(start_url)])
    to_visit = deque([normalize_url(start_url)])
    results = []

    print(f"\n Starting crawl: {start_url}\n")

    while to_visit :
        current_url = to_visit.popleft()

        if current_url in visited:
            continue

        print(f"📄 ({len(visited) + 1}): {current_url}")
        visited.add(current_url)

        text, metadata, soup = get_data_from_website(driver, current_url)

        # Recover from renderer crashes by refreshing the browser instance once.
        if soup is None:
            print("⚠️ Page extraction failed. Reinitializing driver and retrying once...")
            try:
                driver.quit()
            except Exception:
                pass
            driver = create_driver()
            if not driver:
                print("❌ Could not recreate browser driver. Stopping crawl early.")
                break
            text, metadata, soup = get_data_from_website(driver, current_url)

        # Store all pages that have any text content — no minimum character threshold
        if text and text.strip():
            clean_text = text.strip()
            results.append((clean_text, metadata))
            print(f"✅ Stored content: {len(clean_text)} chars")
        else:
            print(f"⚠️ Skipping low content ({len(text.strip()) if text else 0} chars): {current_url}")

        if soup:
            links = get_all_links_sorted(current_url, soup, start_url=start_url)
            new_links = [link for link in links if link not in visited and link not in queued]
            queued.update(new_links)
            to_visit.extend(new_links)
            print(f"   ➕ Added {len(new_links)} new links, queue size: {len(to_visit)}")

        print()

    driver.quit()

    print(f"\n{'=' * 50}")
    print("📊 Crawl Complete:")
    print(f"   Visited: {len(visited)} pages")
    print(f"   Collected: {len(results)} pages with content")
    print(f"{'=' * 50}\n")

    return results