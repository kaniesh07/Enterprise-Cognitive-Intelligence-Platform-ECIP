"""
connectors/web.py
------------------
WebConnector — crawl websites and index their content.

Features:
  - Single URL or full site crawl (follows internal links)
  - Sitemap.xml support (indexes all sitemap URLs at once)
  - robots.txt respect (skips disallowed paths)
  - Configurable depth limit and page count limit
  - Content extracted as clean text via BeautifulSoup
  - Lazy generator — yields one page at a time, O(1) memory

Requires: requests beautifulsoup4
  pip install requests beautifulsoup4

Usage via CLI:
  python scripts/connector_sync.py register --name my_site --type web \
    --config '{"urls":["https://docs.company.com"],"max_depth":3,"max_pages":500}'
"""
from __future__ import annotations

import hashlib
import time
from typing import Generator
from urllib.parse import urljoin, urlparse

import structlog

from connectors.base import BaseConnector, ConnectorDocument

logger = structlog.get_logger()


def _page_title(title: str, url: str) -> str:
    path = urlparse(url).path.strip("/")
    title = (title or "").strip()
    if not path:
        return title or urlparse(url).netloc
    return f"{title} ({path})" if title else path


class WebConnector(BaseConnector):
    """
    Crawls one or more websites and yields their pages as ConnectorDocuments.

    config keys:
      urls        : list[str]  — seed URLs to crawl (required)
      max_depth   : int        — maximum link depth from seed (default 3)
      max_pages   : int        — maximum total pages to crawl (default 200)
      same_domain : bool       — only follow links to the same domain (default True)
      use_sitemap : bool       — try sitemap.xml before crawling (default True)
      respect_robots: bool     — skip URLs blocked by robots.txt (default True)
      include_patterns: list   — only crawl URLs matching these substrings
      exclude_patterns: list   — skip URLs matching these substrings
      delay_seconds: float     — polite delay between requests (default 0.5)
      timeout     : int        — request timeout in seconds (default 15)
      user_agent  : str        — HTTP User-Agent header
    """

    source_type = "web"

    def __init__(self, config: dict):
        super().__init__(config)
        self.seed_urls     = config.get("urls", [])
        self.max_depth     = config.get("max_depth", 3)
        self.max_pages     = config.get("max_pages", 200)
        self.same_domain   = config.get("same_domain", True)
        self.use_sitemap   = config.get("use_sitemap", True)
        self.respect_robots = config.get("respect_robots", True)
        self.include_pats  = config.get("include_patterns", [])
        self.exclude_pats  = config.get("exclude_patterns", [])
        self.delay         = config.get("delay_seconds", 0.5)
        self.timeout       = config.get("timeout", 15)
        self.user_agent    = config.get(
            "user_agent",
            "ECIP-WebConnector/2.0 (enterprise knowledge indexer; respectful crawler)"
        )
        self._session      = None
        self._robots_cache: dict[str, set] = {}   # domain → disallowed paths

    def _get_session(self):
        if self._session is None:
            try:
                import requests
                s = requests.Session()
                s.headers["User-Agent"] = self.user_agent
                s.headers["Accept"] = "text/html,application/xhtml+xml"
                self._session = s
            except ImportError:
                raise ImportError(
                    "requests is required for WebConnector: pip install requests"
                )
        return self._session

    # ── robots.txt ────────────────────────────────────────────────────────

    def _fetch_robots(self, base_url: str) -> set:
        """Return set of disallowed path prefixes from robots.txt."""
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain in self._robots_cache:
            return self._robots_cache[domain]

        disallowed: set = set()
        try:
            r = self._get_session().get(
                f"{domain}/robots.txt", timeout=5, allow_redirects=True
            )
            if r.status_code == 200:
                agent_applies = False
                for line in r.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("user-agent:"):
                        ua = line.split(":", 1)[1].strip()
                        agent_applies = ua in ("*", "ECIP-WebConnector")
                    elif agent_applies and line.lower().startswith("disallow:"):
                        path = line.split(":", 1)[1].strip()
                        if path:
                            disallowed.add(path)
        except Exception:
            pass

        self._robots_cache[domain] = disallowed
        return disallowed

    def _is_allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        disallowed = self._fetch_robots(url)
        path = urlparse(url).path
        return not any(path.startswith(d) for d in disallowed)

    # ── Sitemap ───────────────────────────────────────────────────────────

    def _get_sitemap_urls(self, base_url: str) -> list[str]:
        """Try to fetch and parse sitemap.xml — returns list of URLs."""
        parsed  = urlparse(base_url)
        sitemap = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
        urls    = []
        try:
            r = self._get_session().get(sitemap, timeout=10)
            if r.status_code != 200:
                return []
            import re
            # Extract <loc> URLs from sitemap XML
            locs = re.findall(r"<loc>\s*(https?://[^<\s]+)\s*</loc>",
                              r.text, re.I)
            urls = [u.strip() for u in locs if u.strip()]
            logger.info("sitemap_found", base=base_url, urls=len(urls))
        except Exception as exc:
            logger.debug("sitemap_not_found", base=base_url, error=str(exc)[:60])
        return urls

    # ── Content extraction ────────────────────────────────────────────────

    def _extract_text_and_links(self, html: str, base_url: str
                                ) -> tuple[str, str, list[str]]:
        """
        Extract (title, clean_text, [links]) from HTML.
        Uses BeautifulSoup if available, falls back to regex.
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Page title
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            # Collect links FIRST — menus live in <nav>/<header>/<footer>, which are
            # stripped below (previously the crawler never got past the first page).
            links = []
            for a in soup.find_all("a", href=True):
                href = urljoin(base_url, a["href"])
                if href.startswith("http"):
                    links.append(href)

            # Remove noise tags
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form", "iframe", "noscript"]):
                tag.decompose()

            # Main content heuristic: prefer <main>, <article>, <section>
            main = (soup.find("main") or soup.find("article") or
                    soup.find("section") or soup.find("body") or soup)
            text = main.get_text(separator="\n", strip=True)

            return title, text, links

        except ImportError:
            # Regex fallback (no BeautifulSoup)
            import re
            title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            title   = title_m.group(1).strip() if title_m else ""
            text    = re.sub(r"<[^>]+>", " ", html)
            text    = re.sub(r"\s+", " ", text).strip()
            links   = re.findall(r'href=["\']?(https?://[^"\'>\s]+)', html)
            return title, text, links

    # ── URL filters ───────────────────────────────────────────────────────

    def _should_crawl(self, url: str, seed_domains: set) -> bool:
        """True if URL passes all filters."""
        parsed = urlparse(url)

        # Must be HTTP/HTTPS
        if parsed.scheme not in ("http", "https"):
            return False

        # Skip binary files
        skip_exts = {".pdf", ".docx", ".zip", ".png", ".jpg", ".jpeg",
                     ".gif", ".svg", ".mp4", ".mp3", ".exe", ".dmg"}
        if any(parsed.path.lower().endswith(ext) for ext in skip_exts):
            return False

        # Same-domain filter
        if self.same_domain and parsed.netloc not in seed_domains:
            return False

        # Include patterns
        if self.include_pats and not any(p in url for p in self.include_pats):
            return False

        # Exclude patterns
        if any(p in url for p in self.exclude_pats):
            return False

        # robots.txt
        if not self._is_allowed(url):
            return False

        return True

    # ── Main crawler ──────────────────────────────────────────────────────

    def list(self) -> Generator[ConnectorDocument, None, None]:
        """
        Lazily yield ConnectorDocuments — one per crawled page.
        Memory stays constant regardless of how many pages are crawled.
        """
        if not self.seed_urls:
            return

        session      = self._get_session()
        seed_domains = {urlparse(u).netloc for u in self.seed_urls}
        visited:     set = set()
        # Queue entries: (url, depth)
        queue: list[tuple[str, int]] = []
        total_yielded = 0

        # Seed with sitemap URLs (depth=1) or seed URLs
        for seed in self.seed_urls:
            sitemap_urls = self._get_sitemap_urls(seed) if self.use_sitemap else []
            if sitemap_urls:
                queue.extend((u, 1) for u in sitemap_urls[:self.max_pages])
            else:
                queue.append((seed, 0))

        while queue and total_yielded < self.max_pages:
            url, depth = queue.pop(0)

            # Normalise URL (strip fragment)
            url = url.split("#")[0].rstrip("/")
            if not url or url in visited:
                continue
            if not self._should_crawl(url, seed_domains):
                continue

            visited.add(url)

            try:
                resp = session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    headers={"Accept": "text/html"},
                )
                if resp.status_code != 200:
                    continue
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    continue

                title, text, links = self._extract_text_and_links(resp.text, url)
                if not text.strip():
                    continue

                # Fingerprint based on URL + content hash
                content_hash = hashlib.sha256(resp.text.encode()).hexdigest()[:8]
                source_id    = f"web::{url}"
                fingerprint_val = hashlib.sha256(
                    f"{source_id}:{content_hash}".encode()
                ).hexdigest()[:16]

                doc = ConnectorDocument(
                    source_id     = source_id,
                    source_type   = self.source_type,
                    # Page path keeps titles unique (re-indexing replaces docs by title)
                    title         = _page_title(title, url),
                    content_bytes = text.encode("utf-8"),
                    mime_type     = "text/plain",
                    last_modified = time.time(),
                    size_bytes    = len(resp.content),
                    metadata      = {
                        "url":    url,
                        "domain": urlparse(url).netloc,
                        "depth":  depth,
                        "content_hash": content_hash,
                    },
                )
                # Override fingerprint to use content hash (not last_modified)
                doc._fingerprint_override = fingerprint_val

                yield doc
                total_yielded += 1

                # Queue discovered links
                if depth < self.max_depth:
                    for link in links:
                        link = link.split("#")[0].rstrip("/")
                        if link and link not in visited:
                            queue.append((link, depth + 1))

                if self.delay > 0:
                    time.sleep(self.delay)

            except Exception as exc:
                logger.warning("web_crawl_page_failed", url=url, error=str(exc)[:100])
                continue

        logger.info("web_crawl_done", total=total_yielded, visited=len(visited))

    def load(self, doc: ConnectorDocument) -> bytes:
        """Content is already in content_bytes set during list()."""
        return doc.content_bytes or b""

    def test_connection(self) -> bool:
        if not self.seed_urls:
            return False
        try:
            r = self._get_session().get(self.seed_urls[0], timeout=10)
            return r.status_code == 200
        except Exception:
            return False
