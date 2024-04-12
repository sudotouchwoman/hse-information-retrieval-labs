#! /usr/bin/env python

from dataclasses import dataclass
import logging
import logging.config

from typing import Callable, Mapping, Sequence
import pathlib

import click
import pandas as pd
import yaml

from src.indexer import plain_index


def init_logging():
    with open("config/logging.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.config.dictConfig(cfg)
    return logging.getLogger(__name__)


l = init_logging()


def preview_doc_names(df: pd.DataFrame, doc_ids: Sequence[str]):
    return df.loc[doc_ids, "url"].map(lambda x: x.rpartition("/")[-1])


@dataclass(frozen=True)
class IndexSearcher:
    stemmer: Callable[[str], str]
    index: Mapping[str, plain_index.DocPlainIndex]

    def query(
        self,
        query: str,
    ) -> Sequence[str]:
        words = plain_index.preprocess(query, plain_index.STOP_WORDS, self.stemmer)
        if not words:
            return []

        l.debug(f"running query:  {words}")

        if len(words) == 2:
            l.debug("performing bigram search")
            docs = [
                (k, i.doc_name, i.bigram_mi_score(words))
                for k, i in self.index.items()
                if i.bigram_mi_score(words)
                if not None
            ]

            l.debug("sorting by mutual information")
            docs.sort(key=lambda x: x[-1], reverse=True)

            if docs:
                l.debug(f"MI: max {docs[0][-1]:.4f}, min {docs[-1][-1]:.4f}")

            return [(name, score) for _, name, score in docs]

        # try a simple approach to multi-word search:
        # set intersection
        l.debug("performing simple independent search with set intersection")
        docs = set.intersection(
            *(
                {k for k, i in self.index.items() if i.tf_score(word) is not None}
                for word in words
            )
        )

        return self._rank(words, docs)

    def _rank(self, words: Sequence[str], docs: Sequence[str]):
        if not docs:
            return []

        results = []
        for doc in docs:
            doc_index = self.index[doc]
            score = sum((doc_index.tf_score(w) or 0 for w in words))
            results.append((doc_index.doc_name, score))

        results.sort(key=lambda x: x[-1], reverse=True)
        return results


@click.command()
@click.option("--data-dir", help="Directory to dump data to")
def main(data_dir):
    data_dir = pathlib.Path(data_dir)

    corpus_index = data_dir / "index.json"
    df_corpus_index = pd.read_json(corpus_index, lines=True).set_index("uuid")
    doc_ids = df_corpus_index["url"].map(lambda x: x.rpartition("/")[-1]).to_dict()

    index = plain_index.build_plain_index(data_dir / "pages", doc_ids)
    stemmer = plain_index.default_stemmer()

    searcher = IndexSearcher(stemmer=stemmer, index=index)

    try:
        while True:
            text = input("query> ")
            if not text:
                continue

            if text == r"\q":
                break

            filtered_docs = searcher.query(text)
            l.info(f"found {len(filtered_docs)} documents matching query")
            if filtered_docs:
                l.info(filtered_docs[: min(10, len(filtered_docs))])

    except KeyboardInterrupt:
        l.info("exiting")


if __name__ == "__main__":
    main()
