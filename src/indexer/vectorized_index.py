from dataclasses import dataclass
import pathlib
from typing import Iterable, Sequence, Tuple

import faiss
import sentence_transformers

import numpy as np

from rich import progress

from . import plain_index


@dataclass
class Doc:
    name: str
    doc_uuid: str
    sentences: Sequence[str]


def read_doc(corpus_dir: pathlib.Path, doc_name: str, doc_id: str):
    doc_content = plain_index.read_doc(corpus_dir, doc_id)
    sentences = doc_content.split(".")

    return Doc(name=doc_name, doc_uuid=doc_id, sentences=sentences)


def read_docs(corpus_dir: pathlib.Path, docs: Sequence[Tuple[str, str]]):
    return [read_doc(corpus_dir, doc_name, doc_id) for doc_id, doc_name in docs]


@dataclass
class VectorizedIndex:
    model: sentence_transformers.SentenceTransformer
    ann_index: faiss.Index
    docs: Sequence[Doc]

    def build(self):
        self.ann_index.reset()
        # average embeddings over all sentences for each doc
        doc_embeddings = [
            self.doc_embedding(d)
            for d in progress.track(
                self.docs,
                total=len(self.docs),
                description="Building doc embeddings",
            )
        ]
        doc_embeddings = np.asarray(doc_embeddings)
        doc_embeddings = (
            doc_embeddings / np.linalg.norm(doc_embeddings, axis=1)[:, None]
        )

        self.ann_index.train(doc_embeddings)
        self.ann_index.add(doc_embeddings)

        del doc_embeddings
        return self

    def load(self, p: pathlib.Path):
        self.ann_index = read_index(p)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings

    def doc_embedding(self, doc: Doc):
        return self.encode(doc.sentences).mean(0)

    def query(self, text: str, k: int = 5):
        # assume single sentence as input
        query_vector = self.encode(text)
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = np.expand_dims(query_vector, axis=0)

        distances, indices = self.ann_index.search(query_vector, k=k)
        distances, indices = distances[0], indices[0]

        top_k_docs = [
            (self.docs[i], d)
            for i, d in sorted(
                zip(indices, distances),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        return top_k_docs


def load_embedder(name: str, download=False):
    return sentence_transformers.SentenceTransformer(name, local_files_only=not download)


def create_index(dim: int, num_clusters: int):
    quantizer = faiss.IndexFlatIP(dim)
    ann_index = faiss.IndexIVFFlat(
        quantizer, dim, num_clusters, faiss.METRIC_INNER_PRODUCT
    )
    return ann_index


def read_index(p: pathlib.Path):
    # https://github.com/facebookresearch/faiss/blob/main/tutorial/python/2-IVFFlat.py
    if not p.is_file():
        raise FileNotFoundError(p)

    return faiss.read_index(str(p))


def dump_index(index: faiss.Index, p: pathlib.Path):
    if not index.is_trained:
        raise RuntimeError("index not trained")

    faiss.write_index(index, str(p))
