#! /usr/bin/env python

import logging
import logging.config

import pathlib

import click
import pandas as pd
import yaml

from src.indexer import inverse_index


def init_logging():
    with open("config/logging.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.config.dictConfig(cfg)
    return logging.getLogger(__name__)


l = init_logging()


@click.command()
@click.option("--data-dir", help="Directory to dump data to")
def main(data_dir):
    data_dir = pathlib.Path(data_dir)

    corpus_index = data_dir / "index.json"
    df_corpus_index = pd.read_json(corpus_index, lines=True).set_index("uuid")
    doc_ids = df_corpus_index["url"].map(lambda x: x.rpartition("/")[-1]).to_dict()

    tokenizer = inverse_index.default_tokenizer()

    forward_idx = inverse_index.build_forward_index(
        data_dir / "pages",
        doc_ids,
        tokenizer=tokenizer,
    )

    inverse_idx = inverse_index.build_inverse_index(forward_idx)

    try:
        while True:
            text = input("query> ")
            if not text:
                continue

            if text == r"\q":
                break

            tokens = tokenizer.tokenize(text)
            filtered_docs = inverse_idx.query(tokens)

            l.info(f"found {len(filtered_docs)} documents matching query: {tokens}")
            if filtered_docs:
                reply = "top scores: " + ", ".join(
                    (
                        f"[{title}: {score:.3f}]"
                        for title, score in filtered_docs[: min(10, len(filtered_docs))]
                    )
                )

                l.info(reply)

    except KeyboardInterrupt:
        l.info("exiting")


if __name__ == "__main__":
    main()
