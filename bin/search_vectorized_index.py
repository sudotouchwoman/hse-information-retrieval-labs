#! /usr/bin/env python

import logging
import logging.config

import pathlib

import click
import numpy as np
import pandas as pd
import yaml

from src.indexer import vectorized_index


def init_logging():
    with open("config/logging.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.config.dictConfig(cfg)
    return logging.getLogger(__name__)


l = init_logging()


@click.command()
@click.option("--data-dir", help="Directory to dump data to")
@click.option("--preprocess", help="Build vector index from scratch", type=bool)
@click.option("--max-docs", help="Maximal number of docs to index", type=int)
@click.option("-k", help="Top k docs to query", type=int, default=5)
def main(data_dir, preprocess, max_docs, k):
    data_dir = pathlib.Path(data_dir)

    corpus_index = data_dir / "index.json"
    df_corpus_index = pd.read_json(corpus_index, lines=True).set_index("uuid")
    doc_ids = df_corpus_index["url"].map(lambda x: x.rpartition("/")[-1])
    doc_ids = tuple(doc_ids.items())

    if max_docs:
        doc_ids = doc_ids[:max_docs]

    n = len(doc_ids)

    # NOTE: model should be loaded prior to the test and saved in data_dir:
    # name = "all-MiniLM-L6-v2"
    # model = sentence_transformers.SentenceTransformer(name)
    # model.save(data_dir / "encoders" / name)

    default_model_name = "all-MiniLM-L6-v2"
    model = vectorized_index.load_embedder(str(data_dir / "encoders" / default_model_name))

    dim = model.get_sentence_embedding_dimension()
    num_clusters = int(np.sqrt(n))
    ann_index = vectorized_index.create_index(dim, num_clusters)

    docs = vectorized_index.read_docs(data_dir / "pages", doc_ids)

    index = vectorized_index.VectorizedIndex(
        model=model,
        ann_index=ann_index,
        docs=docs,
    )

    index_path = data_dir / "vector-index.bin"

    if preprocess:
        index.build()
        vectorized_index.dump_index(index.ann_index, index_path)
    else:
        index.load(index_path)

    try:
        while True:
            text = input("query> ")
            if not text:
                continue

            if text == r"\q":
                break

            filtered_docs = index.query(text, k=k)

            l.info(f"found {len(filtered_docs)} similar documents")
            if filtered_docs:
                reply = "top scores: " + ", ".join(
                    (f"[{d.name}: {score:.3f}]" for d, score in filtered_docs)
                )

                l.info(reply)

    except KeyboardInterrupt:
        l.info("exiting")


if __name__ == "__main__":
    main()
