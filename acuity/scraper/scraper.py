"""
ACUITY Framework — Facebook Post Scraper

Navigates to a Facebook community group page and extracts post content
using scroll-and-capture via undetected_chromedriver.

Requires the ``scraper`` extra: ``pip install acuity-framework[scraper]``

Usage:
    >>> from acuity.scraper import FacebookScraper
    >>> scraper = FacebookScraper(output_dir="./data/raw")
    >>> posts = scraper.run(["https://facebook.com/groups/example"])
"""
from __future__ import annotations

import csv
import os
import time
import typing
from dataclasses import dataclass

from .utils import clean_post_text, is_valid_post


@dataclass
class ScraperConfig:
    """Configuration for the Facebook scraper.

    Attributes:
        chrome_user_data_dir: Path to Chrome user data directory for session persistence.
        chrome_version_main: Major Chrome version number for undetected_chromedriver.
        max_posts: Maximum number of posts to collect per URL.
        scroll_delay: Delay in seconds between scroll cycles.
        initial_load_delay: Delay in seconds after page navigation.
        min_post_length: Minimum character length to consider a post.
        output_dir: Directory to save scraped posts CSV.
    """
    chrome_user_data_dir: str = "./data/chrome_session"
    chrome_version_main: int = 149
    max_posts: int = 500
    scroll_delay: int = 4
    initial_load_delay: int = 5
    min_post_length: int = 100
    output_dir: str = "./data/raw"


JS_EXTRACT_TEXT = """
function getFacebookText(el) {
    if (!el) return "";
    return el.innerText.trim();
}

function getAuthorForMessage(msgNode, wrapper) {
    let curr = msgNode;
    let bound = wrapper || document.body;
    while(curr && curr !== bound) {
        let container = curr.parentElement;
        if (container) {
            let candidates = Array.from(container.querySelectorAll('h3, strong, a[role="link"]')).filter(el => {
                return (msgNode.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_PRECEDING);
            });
            if (candidates.length > 0) {
                for (let i = 0; i < candidates.length; i++) {
                    let text = candidates[i].textContent.trim();
                    let lowerText = text.toLowerCase();
                    if (!text || text.length > 60 || lowerText === 'follow' || lowerText === 'reply' || lowerText === 'online status indicator' || lowerText.includes('is in') || /^\\+?\\s*\\d+$/.test(text) || text.length === 1) continue;

                    text = text.split(/\\n|\\s*·\\s*| Follow| is in | is with | is feeling | is at | added a | updated h| shared a | was live| and\\s+\\d+\\s+others| and others/i)[0].trim();
                     if (text && text.length >= 2 && text.length < 50 && !/^\\+?\\s*\\d+$/.test(text)) return text;
                }
            }
        }
        curr = curr.parentElement;
    }
    return "Unknown";
}

let messages = document.querySelectorAll('div[data-ad-preview="message"]');
let posts = [];
let processed = new Set();

for (let i = 0; i < messages.length; i++) {
    if (processed.has(messages[i])) continue;

    let wrapper = messages[i];
    let foundWrapper = false;
    let posIndex = 999999;

    while(wrapper && wrapper.tagName && wrapper.tagName.toLowerCase() === 'div') {
        if (wrapper.hasAttribute('aria-posinset')) {
            foundWrapper = true;
            posIndex = parseInt(wrapper.getAttribute('aria-posinset'), 10) || 999999;
            break;
        }
        wrapper = wrapper.parentElement;
    }

    const getHash = (el) => {
        if (!el || !el.textContent) return "0_";
        return el.textContent.length + "_" + el.textContent.substring(0, 20).replace(/\\s+/g, '');
    };

    if (foundWrapper) {
        let msgsInWrapper = wrapper.querySelectorAll('div[data-ad-preview="message"]');
        let selectedText = "";
        let selectedPoster = "Unknown";
        for(let m of msgsInWrapper) {
            if (!processed.has(m)) {
                processed.add(m);

        let currentHash = getHash(m);
        let cachedHash = m.getAttribute('data-acuity-hash');
        let t = m.getAttribute('data-acuity-text');
        let p = m.getAttribute('data-acuity-poster');

        if (cachedHash !== currentHash || typeof t !== 'string') {
            t = getFacebookText(m);
            m.setAttribute('data-acuity-hash', currentHash);
            m.setAttribute('data-acuity-text', t);

            p = getAuthorForMessage(m, wrapper);
            m.setAttribute('data-acuity-poster', p);
        } else if (!p) {
            p = getAuthorForMessage(m, wrapper);
            m.setAttribute('data-acuity-poster', p);
        }

        if (t) {
            selectedText = t;
            selectedPoster = p;
        }
    }
}
if (selectedText) {
    posts.push({text: selectedText, index: posIndex, poster: selectedPoster, html: selectedPoster === 'Unknown' ? wrapper.outerHTML : ''});
}
    } else {
        processed.add(messages[i]);
        let m = messages[i];

        let currentHash = getHash(m);
        let cachedHash = m.getAttribute('data-acuity-hash');
        let t = m.getAttribute('data-acuity-text');
        let p = m.getAttribute('data-acuity-poster');

        if (cachedHash !== currentHash || typeof t !== 'string') {
            t = getFacebookText(m);
            m.setAttribute('data-acuity-hash', currentHash);
            m.setAttribute('data-acuity-text', t);

            p = getAuthorForMessage(m, m.closest('div.x1yztbdb') || null);
            m.setAttribute('data-acuity-poster', p);
        } else if (!p) {
            p = getAuthorForMessage(m, m.closest('div.x1yztbdb') || null);
            m.setAttribute('data-acuity-poster', p);
        }

        if (t) posts.push({text: t, index: 999999, poster: p, html: p === 'Unknown' ? (m.closest('div.x1yztbdb') || m).outerHTML : ''});
    }
}
return posts;
"""


class FacebookScraper:
    """Scrapes posts from Facebook community group pages.

    Uses ``undetected_chromedriver`` to bypass bot detection and
    scroll-and-capture to extract post content.

    Args:
        config: A ``ScraperConfig`` instance. If ``None``, uses defaults.
    """

    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()

    def run(
        self,
        target_urls: list[str],
        headless: bool = False,
        max_posts: int | None = None,
    ) -> list[dict[str, typing.Any]]:
        """Scrape posts from Facebook page/group URLs.

        Args:
            target_urls: A list of Facebook URLs to scrape.
            headless: Run Chrome in headless mode.
            max_posts: Override the default maximum number of posts per URL.

        Returns:
            List of post dictionaries with keys: ``text``, ``index``,
            ``poster``, ``scraped_at``, ``source_url``.
        """
        try:
            import undetected_chromedriver as uc  # type: ignore
            from selenium.webdriver.common.by import By  # type: ignore
            from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
            from selenium.webdriver.support import expected_conditions as EC  # type: ignore
        except ImportError as e:
            raise ImportError(
                "FacebookScraper requires undetected-chromedriver and selenium. "
                "Install with: pip install acuity-framework[scraper]"
            ) from e
        import urllib.parse

        limit_posts = max_posts if max_posts is not None else self.config.max_posts
        cfg = self.config

        print(f"Using user data directory: {cfg.chrome_user_data_dir}")

        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        if headless:
            options.add_argument("--headless")

        print("Launching undetected Chrome...")
        driver = uc.Chrome(
            options=options,
            user_data_dir=cfg.chrome_user_data_dir,
            version_main=cfg.chrome_version_main,
        )

        all_posts: list[dict[str, typing.Any]] = []

        try:
            time.sleep(2)

            for target_url in target_urls:
                if "/groups/" in target_url and "sorting_setting" not in target_url:
                    parts = urllib.parse.urlparse(target_url)
                    q = urllib.parse.parse_qs(parts.query)
                    q["sorting_setting"] = ["CHRONOLOGICAL"]
                    target_url = parts._replace(query=urllib.parse.urlencode(q, doseq=True)).geturl()
                    print(f"\nAuto-applied 'New Posts' filter. Updated URL: {target_url}")

                print(f"\n--- Navigating to {target_url} ---")
                driver.get(target_url)
                time.sleep(cfg.initial_load_delay)

                if driver.title == "Facebook" and "facebook.com" in driver.current_url:
                    print("Still on homepage. Retrying navigation to target...")
                    driver.get(target_url)
                    time.sleep(cfg.initial_load_delay)

                print(f"Current URL: {driver.current_url}")
                print(f"Page Title:  {driver.title}")

                if "login" in driver.current_url:
                    print("Redirected to login page. Please run login first.")
                    continue

                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except Exception as e:
                    print(f"Timeout waiting for body: {e}")
                    continue

                last_height = driver.execute_script("return document.body.scrollHeight")
                print(f"Starting scrape for {target_url}...")

                posts: list[dict[str, typing.Any]] = []
                scroll_idx = 0
                while True:
                    scroll_idx += 1
                    print(f"Starting scroll cycle {scroll_idx}")

                    steps_per_cycle = 6
                    for step in range(steps_per_cycle):
                        driver.execute_script("window.scrollBy(0, 600);")
                        time.sleep(0.8)

                        driver.execute_script("""
                            document.querySelectorAll('div[role="button"]:not([data-acuity-checked="true"])').forEach(btn => {
                                let text = btn.textContent;
                                if (text && text.toLowerCase().includes('see more')) {
                                    try { btn.click(); btn.setAttribute('data-acuity-checked', 'true'); } catch(e) {}
                                } else {
                                    btn.setAttribute('data-acuity-checked', 'true');
                                }
                            });
                        """)
                        time.sleep(0.4)

                        extracted_data = driver.execute_script(JS_EXTRACT_TEXT)

                        for item in extracted_data:
                            raw_text = item.get("text", "")
                            index = item.get("index", 999999)
                            poster = item.get("poster", "Unknown")

                            try:
                                text = clean_post_text(raw_text)
                                if is_valid_post(text, min_length=cfg.min_post_length):
                                    existing_post = next(
                                        (p for p in posts if (p["index"] == index and index != 999999) or p["text"] == text),
                                        None,
                                    )

                                    if existing_post:
                                        if len(text) > len(existing_post["text"]):
                                            existing_post["text"] = text
                                            existing_post["poster"] = poster
                                    else:
                                        posts.append({
                                            "text": text,
                                            "index": index,
                                            "poster": poster,
                                            "scraped_at": time.time(),
                                            "source_url": target_url,
                                        })
                                        display_text = text[:50].replace("\n", " ").encode("ascii", "replace").decode("ascii")
                                        print(f"  Captured (Index {index}) by {poster}: {display_text}...")
                            except Exception as e:
                                print(f"  Error extracting post: {e}")
                                continue

                    print(f"Finished cycle {scroll_idx}. Total unique posts so far: {len(posts)}")
                    time.sleep(cfg.scroll_delay / 2)

                    if len(posts) >= limit_posts:
                        print(f"Reached target of {limit_posts} posts. Stopping early.")
                        break

                    posts.sort(key=lambda x: x["index"])

                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        time.sleep(2)
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        if new_height == last_height:
                            print("End of content or stuck. Stopping.")
                            break
                    last_height = new_height

                while len(posts) > limit_posts:
                    posts.pop()

                all_posts.extend(posts)
                print(f"Finished scraping {target_url}. Gathered {len(posts)} posts.\n")

        except Exception as e:
            error_str = str(e)
            if "invalid session id" in error_str or "disconnected" in error_str:
                print("\nBrowser was closed manually. Stopping scrape.")
            else:
                print(f"\nError during scraping: {e}")
            print("Scraping interrupted. Evaluating partial data to save...")
        finally:
            # Save results
            try:
                os.makedirs(cfg.output_dir, exist_ok=True)
                output_file = os.path.join(cfg.output_dir, "posts.csv")

                if all_posts:
                    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
                        writer = csv.DictWriter(f, fieldnames=["index", "poster", "text", "scraped_at", "source_url"])
                        writer.writeheader()
                        writer.writerows(all_posts)
                    print(f"Saved {len(all_posts)} posts to {output_file}")
                else:
                    print("No posts found.")
            except Exception as save_err:
                print(f"Failed to save results: {save_err}")

            try:
                driver.quit()
            except Exception:
                pass

        return all_posts
