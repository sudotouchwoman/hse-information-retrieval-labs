#! /usr/bin/env python

import logging
import logging.config
import pathlib
import threading
import click
import yaml

from src.scraper import scraper


def init_logging():
    with open("config/logging.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.config.dictConfig(cfg)
    return logging.getLogger(__name__)


@click.command()
@click.option("--max-words", default=100, help="Total number of words to scrape")
@click.option("--data-dir", help="Directory to dump data to")
def main(max_words, data_dir):
    log = init_logging()

    wikihow_url = "https://www.wikihow.com"
    random_page_url = "/Special:Randomizer"

    loop = scraper.ProcessingLoop(
        base_url=wikihow_url,
        random_page_url=random_page_url,
        max_words=max_words,
    )

    # put something into the queue prior to processing
    # to avoid deadlocking on reads
    loop.pending_tasks.put(scraper.PrioritizedItem(random_page_url))

    data_dir = pathlib.Path(data_dir).resolve()
    pages_dir = data_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"will dump pages to {pages_dir}")

    threading.Thread(
        target=loop.process_pages,
        name="process-pages",
        daemon=True,
    ).start()

    with open(data_dir / "index.json", "w", encoding="utf-8") as index:
        loop.dump_results(index, pages_dir)

    log.info("processing completed")


if __name__ == "__main__":
    main()
