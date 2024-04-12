import collections
from dataclasses import dataclass
import math
import pathlib
import re
from typing import Callable, FrozenSet, Mapping, Sequence, Set

from rich import progress
from urllib import parse

from . import plain_index


@dataclass(frozen=True)
class Doc:
    name: str
    doc_uuid: str
    words: Mapping[str, float]
    word_count: int

    def tf(self, word: str):
        return self.words.get(word, 0) / self.word_count


@dataclass(frozen=True)
class Tokenizer:
    stemmer_func: Callable[[str], str]
    stopwords: Set[str]
    word_pattern: re.Pattern

    def tokenize(self, text: str):
        words = (x.group() for x in self.word_pattern.finditer(text.lower()))
        words = (w for w in words if w not in self.stopwords)
        words = (self.stemmer_func(w) for w in words)

        return tuple(w for w in words if w)


def default_tokenizer():
    return Tokenizer(
        stemmer_func=plain_index.default_stemmer().stem,
        stopwords=plain_index.STOP_WORDS,
        word_pattern=plain_index.WORD_PATTERN,
    )


def build_forward_index(
    corpus_dir: pathlib.Path,
    doc_ids: Mapping[str, str],
    tokenizer: Tokenizer,
):
    # creates plain (forward) index from given
    # text corpus

    def build_doc_words(doc_id: str, doc_name: str):
        doc_content = plain_index.read_doc(corpus_dir, doc_id)
        tokens = tokenizer.tokenize(doc_content)

        return Doc(
            name=parse.unquote(doc_name),
            doc_uuid=doc_id,
            words=collections.Counter(tokens),
            word_count=len(tokens),
        )

    forward_index = {
        doc_id: build_doc_words(doc_id, doc_name)
        for doc_id, doc_name in progress.track(
            doc_ids.items(),
            total=len(doc_ids),
            description="Building forward index",
        )
    }

    return forward_index


@dataclass(frozen=True)
class InverseIndex:
    inverse_mapping: Mapping[str, FrozenSet[str]]
    forward_mapping: Mapping[str, Doc]
    idf_index: Mapping[str, float]

    def query(self, tokens: Sequence[str]):
        # find documents containing tokens using
        # inverse index, then compute tf-idf scores for each
        # token in query and perform simple ranking
        matching_docs = set.intersection(
            *(self.inverse_mapping.get(t, set()) for t in tokens)
        )

        matching_docs = (self.forward_mapping[doc] for doc in matching_docs)
        return self.__rank_docs(matching_docs, tokens)

    def __rank_docs(self, docs: Sequence[Doc], tokens: Sequence[str]):
        # compute cumulative doc scores based on token tf-idf scores
        doc_scores = [
            (doc.name, sum(doc.tf(t) * self.idf_index.get(t, 0) for t in tokens))
            for doc in docs
        ]

        # rank docs by tf-idf scores for query
        doc_scores.sort(key=lambda x: x[-1], reverse=True)
        return doc_scores


def build_inverse_index(forward_index: Mapping[str, Doc]) -> InverseIndex:
    # for each word, get number of documents it is present in the dataset
    inverse_index = collections.defaultdict(set)

    for doc_id, doc in progress.track(
        forward_index.items(),
        total=len(forward_index),
        description="Building inverse index",
    ):
        for word in doc.words:
            inverse_index[word].add(doc_id)

    # then, compute idf for each term and cache it for
    # upcoming queries

    total_docs = len(forward_index)
    idf_index = {
        word: math.log10(total_docs / len(docs)) for word, docs in inverse_index.items()
    }

    return InverseIndex(
        inverse_mapping=inverse_index,
        forward_mapping=forward_index,
        idf_index=idf_index,
    )
