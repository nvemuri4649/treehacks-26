"""
Web search engine using Bright Data APIs.

Provides three search strategies:
1. SERP text/image search via Bright Data SDK
2. Reverse image search via Bright Data Scraping Browser (Google Lens)
3. Platform-specific crawls via Scraping Browser
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus, urljoin

import httpx
from playwright.async_api import async_playwright, Page, Browser

from deepfake.core.config import settings

logger = logging.getLogger(__name__)


#---------------------------------------------------------------------------
#Data models
#---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single text search result."""

    title: str
    url: str
    snippet: str
    position: int = 0


@dataclass
class ImageResult:
    """A single image search result."""

    image_url: str
    thumbnail_url: str = ""
    source_url: str = ""
    title: str = ""
    width: int = 0
    height: int = 0


@dataclass
class ReverseSearchResult:
    """A result from reverse image search."""

    page_url: str
    image_url: str = ""
    title: str = ""
    snippet: str = ""
    similarity_label: str = ""  # e.g. "visually similar", "exact match"


#---------------------------------------------------------------------------
#SERP Search (Bright Data SDK / REST API)
#---------------------------------------------------------------------------

class SerpSearchEngine:
    """Text and image search using Bright Data's SERP API."""

    BASE_URL = "https://api.brightdata.com/request"

    def __init__(self):
        self._token = settings.brightdata_api_token
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._token}",
                },
            )
        return self._client

    async def search_text(
        self,
        query: str,
        num_results: int = 20,
        country: str = "us",
        language: str = "en",
    ) -> list[SearchResult]:
        """
        Perform a Google text search via Bright Data SERP API.

        Returns parsed search results with title, URL, and snippet.
        """
        logger.info("SERP text search: '%s' (n=%d)", query, num_results)
        client = await self._get_client()

        encoded_query = quote_plus(query)
        search_url = (
            f"https://www.google.com/search?q={encoded_query}"
            f"&hl={language}&gl={country}&num={num_results}"
        )

        try:
            response = await client.post(
                self.BASE_URL,
                json={
                    "zone": "serp",
                    "url": search_url,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("SERP text search failed: %s", e)
            return []

        results = []
        organic = data.get("organic", data.get("results", []))
        for idx, item in enumerate(organic[:num_results]):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", item.get("url", "")),
                    snippet=item.get("snippet", item.get("description", "")),
                    position=idx + 1,
                )
            )

        logger.info("SERP text search returned %d results", len(results))
        return results

    async def search_images(
        self,
        query: str,
        num_results: int = 30,
        country: str = "us",
        language: str = "en",
    ) -> list[ImageResult]:
        """
        Perform a Google Images search via Bright Data SERP API.

        Returns image URLs with metadata.
        """
        logger.info("SERP image search: '%s' (n=%d)", query, num_results)
        client = await self._get_client()

        encoded_query = quote_plus(query)
        search_url = (
            f"https://www.google.com/search?q={encoded_query}"
            f"&tbm=isch&hl={language}&gl={country}"
        )

        try:
            response = await client.post(
                self.BASE_URL,
                json={
                    "zone": "serp",
                    "url": search_url,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("SERP image search failed: %s", e)
            return []

        results = []
        images = data.get("image_results", data.get("images", []))
        for item in images[:num_results]:
            results.append(
                ImageResult(
                    image_url=item.get("original", item.get("image_url", "")),
                    thumbnail_url=item.get("thumbnail", item.get("thumbnail_url", "")),
                    source_url=item.get("source", item.get("source_url", "")),
                    title=item.get("title", ""),
                    width=item.get("width", 0),
                    height=item.get("height", 0),
                )
            )

        logger.info("SERP image search returned %d results", len(results))
        return results

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


#---------------------------------------------------------------------------
#Reverse Image Search (Bright Data Scraping Browser + Google Lens)
#---------------------------------------------------------------------------

class ReverseImageSearchEngine:
    """
    Reverse image search using Bright Data's Scraping Browser.

    Automates Google Lens via Playwright over CDP to find visually similar
    images across the web.
    """

    def __init__(self):
        self._auth = settings.brightdata_browser_auth
        self._browser: Optional[Browser] = None

    def _get_endpoint(self) -> str:
        return f"wss://{self._auth}@brd.superproxy.io:9222"

    async def reverse_image_search(
        self,
        image_path: str | Path,
        max_results: int = 30,
    ) -> list[ReverseSearchResult]:
        """
        Perform a reverse image search using Google Lens via Bright Data Scraping Browser.

        Uploads the image to Google Lens, waits for results, then parses
        visually similar images.

        Args:
            image_path: local path to the image file.
            max_results: maximum results to return.

        Returns:
            List of ReverseSearchResult with page/image URLs.
        """
        logger.info("Reverse image search for: %s", image_path)
        results: list[ReverseSearchResult] = []

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(self._get_endpoint())
                try:
                    context = browser.contexts[0] if browser.contexts else await browser.new_context()
                    page = await context.new_page()

                    #Navigate to Google Images
                    await page.goto("https://images.google.com", timeout=60_000)
                    await page.wait_for_load_state("domcontentloaded")

                    #Click the camera/lens icon to open reverse image search
                    lens_button = page.locator('[aria-label="Search by image"]').first
                    await lens_button.click(timeout=10_000)
                    await asyncio.sleep(1.5)

                    #Click "upload a file" tab/link
                    upload_tab = page.get_by_text("upload a file", exact=False).first
                    await upload_tab.click(timeout=10_000)
                    await asyncio.sleep(1)

                    #Upload the image file
                    file_input = page.locator('input[type="file"]').first
                    await file_input.set_input_files(str(image_path))

                    #Wait for Google Lens results to load
                    await page.wait_for_load_state("networkidle", timeout=30_000)
                    await asyncio.sleep(3)

                    #Parse results - Google Lens shows visual matches
                    results = await self._parse_lens_results(page, max_results)

                    logger.info("Reverse image search found %d results", len(results))

                except Exception as e:
                    logger.error("Reverse image search browser error: %s", e)
                finally:
                    await browser.close()

        except Exception as e:
            logger.error("Failed to connect to Scraping Browser: %s", e)

        return results

    async def _parse_lens_results(
        self, page: Page, max_results: int
    ) -> list[ReverseSearchResult]:
        """Parse Google Lens results page for visually similar images."""
        results: list[ReverseSearchResult] = []

        try:
            #Try to find the "Find image source" or visual matches section
            #Google Lens renders results as cards with links
            await page.wait_for_selector('a[href*="http"]', timeout=15_000)

            #Extract all result links with images
            items = await page.evaluate("""() => {
                const results = [];
                // Look for result cards/links in the visual matches area
                const links = document.querySelectorAll('a[href^="http"]');
                const seen = new Set();
                
                for (const link of links) {
                    const href = link.href;
                    if (seen.has(href) || href.includes('google.com') || href.includes('gstatic.com')) {
                        continue;
                    }
                    seen.add(href);
                    
                    const img = link.querySelector('img');
                    const title = link.textContent?.trim().slice(0, 200) || '';
                    const imgSrc = img ? (img.src || img.dataset.src || '') : '';
                    
                    if (href && title) {
                        results.push({
                            page_url: href,
                            image_url: imgSrc,
                            title: title,
                        });
                    }
                }
                return results;
            }""")

            for item in items[:max_results]:
                results.append(
                    ReverseSearchResult(
                        page_url=item.get("page_url", ""),
                        image_url=item.get("image_url", ""),
                        title=item.get("title", ""),
                        similarity_label="visually similar",
                    )
                )

        except Exception as e:
            logger.warning("Failed to parse Lens results: %s", e)

            #Fallback: try getting page text for any useful info
            try:
                content = await page.content()
                urls = re.findall(r'https?://(?!.*google\.com)[^\s"<>]+', content)
                seen = set()
                for url in urls[:max_results]:
                    clean = url.rstrip("',;)")
                    if clean not in seen:
                        seen.add(clean)
                        results.append(
                            ReverseSearchResult(
                                page_url=clean,
                                title="",
                                similarity_label="extracted",
                            )
                        )
            except Exception:
                pass

        return results

    async def search_by_url(
        self,
        image_url: str,
        max_results: int = 30,
    ) -> list[ReverseSearchResult]:
        """
        Reverse image search using an image URL (without file upload).

        Uses the Google searchbyimage endpoint.
        """
        logger.info("Reverse search by URL: %s", image_url[:80])
        results: list[ReverseSearchResult] = []

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(self._get_endpoint())
                try:
                    context = browser.contexts[0] if browser.contexts else await browser.new_context()
                    page = await context.new_page()

                    encoded_url = quote_plus(image_url)
                    search_url = (
                        f"https://www.google.com/searchbyimage?image_url={encoded_url}"
                    )
                    await page.goto(search_url, timeout=60_000)
                    await page.wait_for_load_state("networkidle", timeout=30_000)
                    await asyncio.sleep(2)

                    results = await self._parse_lens_results(page, max_results)

                except Exception as e:
                    logger.error("Reverse search by URL browser error: %s", e)
                finally:
                    await browser.close()

        except Exception as e:
            logger.error("Failed to connect to Scraping Browser: %s", e)

        return results


#---------------------------------------------------------------------------
#Platform-Specific Crawls
#---------------------------------------------------------------------------

class PlatformCrawler:
    """
    Targeted crawls on specific platforms where deepfakes are commonly shared.

    Uses Bright Data Scraping Browser to bypass access restrictions.
    """

    PLATFORM_SEARCH_URLS = {
        "reddit": "https://www.reddit.com/search/?q={query}&type=link",
        "twitter": "https://x.com/search?q={query}&f=media",
    }

    def __init__(self):
        self._auth = settings.brightdata_browser_auth

    def _get_endpoint(self) -> str:
        return f"wss://{self._auth}@brd.superproxy.io:9222"

    async def crawl_platform(
        self,
        platform: str,
        query: str,
        max_results: int = 20,
    ) -> list[SearchResult]:
        """
        Search a specific platform for deepfake content.

        Args:
            platform: platform name (reddit, twitter).
            query: search query string.
            max_results: maximum results to return.

        Returns:
            List of SearchResult from the platform.
        """
        if platform not in self.PLATFORM_SEARCH_URLS:
            logger.warning("Unknown platform: %s", platform)
            return []

        url_template = self.PLATFORM_SEARCH_URLS[platform]
        search_url = url_template.format(query=quote_plus(query))
        logger.info("Platform crawl [%s]: '%s'", platform, query)

        results: list[SearchResult] = []

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(self._get_endpoint())
                try:
                    context = browser.contexts[0] if browser.contexts else await browser.new_context()
                    page = await context.new_page()
                    await page.goto(search_url, timeout=60_000)
                    await page.wait_for_load_state("networkidle", timeout=30_000)
                    await asyncio.sleep(3)

                    results = await self._extract_platform_results(
                        page, platform, max_results
                    )
                except Exception as e:
                    logger.error("Platform crawl [%s] error: %s", platform, e)
                finally:
                    await browser.close()

        except Exception as e:
            logger.error("Failed to connect for platform crawl: %s", e)

        logger.info("Platform crawl [%s] returned %d results", platform, len(results))
        return results

    async def _extract_platform_results(
        self, page: Page, platform: str, max_results: int
    ) -> list[SearchResult]:
        """Extract search results from a platform page."""
        items = await page.evaluate("""() => {
            const results = [];
            const links = document.querySelectorAll('a[href^="http"]');
            const seen = new Set();
            
            for (const link of links) {
                const href = link.href;
                if (seen.has(href)) continue;
                seen.add(href);
                
                const text = link.textContent?.trim() || '';
                if (text.length > 10 && text.length < 500) {
                    results.push({
                        title: text.slice(0, 200),
                        url: href,
                        snippet: text.slice(0, 300),
                    });
                }
            }
            return results;
        }""")

        results = []
        for idx, item in enumerate(items[:max_results]):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    position=idx + 1,
                )
            )
        return results


#---------------------------------------------------------------------------
#Unified Search Facade
#---------------------------------------------------------------------------

class SearchEngine:
    """
    Unified search facade combining SERP, reverse image, and platform searches.

    Manages lifecycle of underlying engines.
    """

    def __init__(self):
        self.serp = SerpSearchEngine()
        self.reverse = ReverseImageSearchEngine()
        self.platforms = PlatformCrawler()

    async def fan_out_search(
        self,
        name: str,
        queries: list[str] | None = None,
        image_path: str | Path | None = None,
        platforms: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute multiple search strategies concurrently.

        Args:
            name: identified name of the person (can be empty).
            queries: explicit text queries to search.
            image_path: path to image for reverse search.
            platforms: platform names for targeted crawls.

        Returns:
            Dict with keys 'text_results', 'image_results',
            'reverse_results', 'platform_results'.
        """
        tasks = {}

        #Default deepfake-oriented queries if name is known
        if not queries and name:
            queries = [
                f'"{name}" deepfake',
                f'"{name}" AI generated face',
                f'"{name}" face swap',
            ]

        #SERP text + image searches
        if queries:
            for i, q in enumerate(queries):
                tasks[f"text_{i}"] = self.serp.search_text(q)
                tasks[f"images_{i}"] = self.serp.search_images(q)

        #Reverse image search
        if image_path:
            tasks["reverse"] = self.reverse.reverse_image_search(image_path)

        #Platform crawls
        if platforms and name:
            for platform in platforms:
                deepfake_query = f"{name} deepfake OR AI generated OR face swap"
                tasks[f"platform_{platform}"] = self.platforms.crawl_platform(
                    platform, deepfake_query
                )

        #Execute all searches concurrently
        if not tasks:
            return {
                "text_results": [],
                "image_results": [],
                "reverse_results": [],
                "platform_results": [],
            }

        keys = list(tasks.keys())
        coros = list(tasks.values())
        results_list = await asyncio.gather(*coros, return_exceptions=True)

        text_results: list[SearchResult] = []
        image_results: list[ImageResult] = []
        reverse_results: list[ReverseSearchResult] = []
        platform_results: list[SearchResult] = []

        for key, result in zip(keys, results_list):
            if isinstance(result, Exception):
                logger.error("Search task '%s' failed: %s", key, result)
                continue
            if key.startswith("text_"):
                text_results.extend(result)
            elif key.startswith("images_"):
                image_results.extend(result)
            elif key == "reverse":
                reverse_results.extend(result)
            elif key.startswith("platform_"):
                platform_results.extend(result)

        return {
            "text_results": text_results,
            "image_results": image_results,
            "reverse_results": reverse_results,
            "platform_results": platform_results,
        }

    async def close(self):
        await self.serp.close()
