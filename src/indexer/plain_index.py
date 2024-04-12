import math
import pathlib
import re
import collections
import itertools

from dataclasses import dataclass
from typing import Mapping, Sequence, Set, Tuple

from rich import progress
from nltk import stem

from urllib import parse


WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")
STOP_WORDS_STRICT = {
    # articles
    "a",
    "an",
    "the",
    # personals
    "i",
    "me",
    "you",
    "your",
    "it",
    "he",
    "she",
    "this",
    "that",
    # proposals
    "to",
    "of",
    "in",
    "at",
    "by",
    "from",
    "out",
    "on",
    # common verbs
    "be",
    "is",
    "was",
    "were",
    "have",
    "been",
    # conjuctions
    "and",
    "or",
    # other
    "if",
    "also",
    "for",
    # modals
    "can",
    "could",
}

STOP_WORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
}


def read_doc(corpus_dir: pathlib.Path, doc_id: str):
    with open(corpus_dir / f"{doc_id}.txt", encoding="utf-8") as f:
        return f.read()


def preprocess(text: str, stopwords: Set[str], stemmer):
    words = filter(
        lambda x: x not in stopwords,
        map(
            lambda x: x.group(),
            WORD_PATTERN.finditer(
                text.lower(),
            ),
        ),
    )

    words = map(stemmer.stem, words)
    return tuple(words)


def build_collocations(words: Sequence[str]):
    # build unigram index
    unigrams = collections.defaultdict(int)
    for w in words:
        unigrams[w] += 1

    # convert unigram counts to frequencies
    unigrams = {k: v / len(unigrams) for k, v in unigrams.items() if v > 2}

    # build bigram index
    bigrams = collections.defaultdict(int)
    for word, collocant in itertools.pairwise(words):
        bigrams[(word, collocant)] += 1

    # convert bigram counts to frequencies
    bigrams = {k: v / len(bigrams) for k, v in bigrams.items() if v > 2}

    def mutual_information(pair):
        x, y = pair
        return math.log(bigrams[pair] / unigrams[x] / unigrams[y])

    bigram_collocations = {b: mutual_information(b) for b in bigrams}
    return unigrams, bigram_collocations


@dataclass(frozen=True, slots=True)
class DocPlainIndex:
    doc_id: str
    doc_name: str
    unigrams: Mapping[str, float]
    bigrams: Mapping[Tuple[str, str], float]

    def tf_score(self, term: str):
        return self.unigrams.get(term, None)

    def bigram_mi_score(self, bigram: Tuple[str, str]):
        return self.bigrams.get(bigram, None)


def default_stemmer():
    return stem.SnowballStemmer("english")


def build_plain_index(corpus_dir: pathlib.Path, doc_ids: Mapping[str, str]):
    stemmer = default_stemmer()

    def doc_words(doc_id: str, doc_name: str):
        doc_content = read_doc(corpus_dir, doc_id)

        words = preprocess(
            doc_content,
            STOP_WORDS,
            stemmer,
        )

        unigrams, bigrams = build_collocations(words)

        return DocPlainIndex(
            doc_id,
            doc_name=parse.unquote(doc_name),
            unigrams=unigrams,
            bigrams=bigrams,
        )

    return {
        d: words
        for d, words in progress.track(
            ((d, doc_words(d, name)) for d, name in doc_ids.items()),
            description="Building plain doc index",
            total=len(doc_ids),
            show_speed=True,
        )
    }
