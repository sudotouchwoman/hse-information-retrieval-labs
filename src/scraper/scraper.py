from dataclasses import dataclass, field
import datetime
import functools
import io
import logging
import pathlib
import queue
import time
from typing import Iterable, Mapping, Protocol, Sequence
from lxml import html
import requests

import ujson
import uuid

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PageMetadata:
    url: str
    text: str
    num_words: int
    num_sentences: int
    featured_pages: Sequence[str]

    def as_dict(self):
        return {
            "url": self.url,
            "num_words": self.num_words,
            "num_sentences": self.num_sentences,
            "featured_pages": self.featured_pages,
        }


class PageParser(Protocol):
    def parse(self, url: str, text: str) -> PageMetadata: ...


class Metrics(Protocol):
    def request_time(self, elapsed: float): ...

    def tick_errors(self): ...


class LogMetrics(Metrics):
    def request_time(self, elapsed: datetime.timedelta):
        log.debug(f"request time: {elapsed.total_seconds():.3f}s")


@dataclass(slots=True, frozen=True)
class UrlFetcher:
    session: requests.Session
    base_url: str
    parser: PageParser
    metrics: Metrics

    def fetch(self, page: str) -> PageMetadata | None:
        url = self.base_url + page
        log.debug(f"fetching {url}")

        with self.session.get(url=url) as response:
            self.metrics.request_time(response.elapsed)
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                log.error(f"while getting {url}: {e}")
                self.metrics.tick_errors()
                return

            text = response.text
            url = response.url

        log.debug(f"fetched {url}, parsing")
        return self.parser.parse(url, text)


@dataclass(slots=True, frozen=True)
class XPathPageParser(PageParser):
    text_query: str
    featured_pages_query: str

    def __xpath_match(self, tree, query: str) -> Iterable[str]:
        # returns all non-empty matches of xpath query
        return filter(
            lambda x: x and len(x) > 1,
            map(
                lambda x: x.strip() if x else None,
                tree.xpath(query),
            ),
        )

    def parse(self, url: str, text: str) -> PageMetadata:
        log.debug(f"parsing {url}: {len(text)}")
        tree = html.fromstring(text)

        text = " ".join(self.__xpath_match(tree, self.text_query))
        num_words = len(text.split())
        num_sentences = len(text.split("."))
        pages = tuple(self.__xpath_match(tree, self.featured_pages_query))

        return PageMetadata(
            url=url,
            text=text,
            num_words=num_words,
            num_sentences=num_sentences,
            featured_pages=pages,
        )


@dataclass(order=True)
class PrioritizedItem:
    item: int | PageMetadata = field(compare=False)
    priority: int = 0


@dataclass(slots=True)
class ProcessingLoop:
    base_url: str
    random_page_url: str
    max_words: int
    total_words: int = 0
    pending_tasks: queue.PriorityQueue[PrioritizedItem] = field(
        default_factory=queue.PriorityQueue
    )
    completed_tasks: queue.PriorityQueue[PrioritizedItem] = field(
        default_factory=functools.partial(queue.PriorityQueue, maxsize=1)
    )
    url_cache: Mapping[str, PageMetadata] = field(default_factory=dict)

    def save_page(self, page: PageMetadata):
        log.debug(f"saving page: {page.url}")

        if page.url in self.url_cache:
            log.warning(f"duplicate page found: {page.url}")
            self.pending_tasks.put(PrioritizedItem(self.random_page_url))
            return

        self.url_cache[page.url] = page
        self.completed_tasks.put(PrioritizedItem(page))
        log.debug(f"pushed {page.url} to completed queue")

        for unvisited_page in filter(
            lambda p: (self.base_url + p) not in self.url_cache, page.featured_pages
        ):
            log.debug(f"to pending tasks: {unvisited_page}")
            self.pending_tasks.put(PrioritizedItem(unvisited_page))

        log.debug("to pending tasks: random page")
        self.pending_tasks.put(PrioritizedItem(self.random_page_url))

    @property
    def text_query(self):
        return "|".join(
            (
                "//div[@class='step']/text()",
                "//div[@class='step']/b/text()",
                "//div[@class='step']/ul//li/text()",
            )
        )

    @property
    def featured_pages_query(self):
        return "//a[@class='related-wh']/@href"

    def process_pages(self):
        log.info("start processing tasks")
        parser = XPathPageParser(self.text_query, self.featured_pages_query)

        with requests.Session() as sess:
            fetcher = UrlFetcher(
                session=sess,
                base_url=self.base_url,
                parser=parser,
                metrics=LogMetrics(),
            )

            for item in iter(self.pending_tasks.get, None):
                task = item.item
                if not task:
                    log.info("stop processing tasks")
                    return

                log.debug(f"picked up task: {task}")
                page = fetcher.fetch(task)
                log.debug("sleeping for a bit not to DDoS the target")
                time.sleep(1e-2)

                if not page:
                    log.warning(f"{task} did not fetch")
                    continue

                self.save_page(page)

            # push a sentinel
            self.completed_tasks.put(PrioritizedItem(None, priority=1))
            log.info("to completed tasks: push sentinel")

    def dump_results(self, pages_index_file: io.FileIO, root_dir: pathlib.Path):
        log.info("start dumping results to filesystem")

        for item in iter(self.completed_tasks.get, None):
            log.debug(f"got completed task")
            page = item.item
            if not page:
                log.info("stop dumping results")
                return

            page_id = uuid.uuid5(uuid.NAMESPACE_URL, page.url).hex
            page_metadata = page.as_dict()
            page_metadata["uuid"] = page_id

            log.debug(f"dumping page metadata to index: {page_id}")
            ujson.dump(page_metadata, pages_index_file, sort_keys=True, indent=0)
            pages_index_file.write("\n")

            log.debug(f"dumping page text to file: {page_id}")
            with open(root_dir / f"{page_id}.txt", "w", encoding="utf-8") as f:
                f.write(page.text)

            self.total_words += page.num_words
            if self.total_words < self.max_words:
                continue

            log.info(f"collected {self.total_words} words, stopping")

            # push a sentinel
            self.pending_tasks.put(PrioritizedItem(None, priority=1))
            break

        log.info("stop dumping results to filesystem")
