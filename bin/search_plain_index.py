#! /usr/bin/env python

import logging
import logging.config

from typing import Sequence
import pathlib

import click
import pandas as pd
from rich import progress
import yaml

from src.indexer import plain_index


def init_logging():
    with open("config/logging.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.config.dictConfig(cfg)
    return logging.getLogger(__name__)


def build_plain_index(corpus_dir: pathlib.Path, doc_ids: Sequence[str]):
    def doc_words(doc_id: str):
        doc_content = plain_index.read_doc(corpus_dir, doc_id)
        unigrams, bigrams = plain_index.build_collocations(doc_content)
        return plain_index.DocPlainIndex(doc_id, unigrams=unigrams, bigrams=bigrams)

    return tuple(
        progress.track(
            (doc_words(d) for d in doc_ids),
            description="Building plain doc index",
            total=len(doc_ids),
        )
    )


def preview_doc_names(df: pd.DataFrame, doc_ids: Sequence[str]):
    return df[df["uuid"].isin(doc_ids)]["url"].map(lambda x: x.rpartition("/")[-1])


def search_plain_index(
    idx: Sequence[plain_index.DocPlainIndex],
    query: str,
) -> Sequence[str]:
    words = tuple(query.split())

    if len(words) > 2:
        raise ValueError("only single-word and bigram search is supported")

    if len(words) == 2:
        return [i.doc_id for i in idx if i.has_bigram(words)]

    (word,) = words
    return [i.doc_id for i in idx if i.has_word(word)]


@click.command()
@click.option("--data-dir", help="Directory to dump data to")
def main(data_dir):
    data_dir = pathlib.Path(data_dir)

    l = init_logging()

    corpus_index = data_dir / "index.json"
    df_corpus_index = pd.read_json(corpus_index, lines=True)
    doc_ids = df_corpus_index["uuid"].to_list()

    plain_index = build_plain_index(data_dir / "pages", doc_ids)

    try:
        while True:
            text = input("query> ")
            if not text:
                continue

            if text == r"\q":
                break

            ids = search_plain_index(plain_index, text)
            l.info(f"found {len(ids)} documents matching query")
            if ids:
                l.info(
                    preview_doc_names(df_corpus_index, ids)
                    .head(min(10, len(ids)))
                    .tolist()
                )

    except KeyboardInterrupt:
        l.info("exiting")


if __name__ == "__main__":
    main()
