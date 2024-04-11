import pathlib
import re
import collections
import itertools

from dataclasses import dataclass
from typing import Mapping, Sequence, Set, Tuple


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
    # articles
    "a",
    "an",
    "the",
}


def read_doc(corpus_dir: pathlib.Path, doc_id: str):
    with open(corpus_dir / f"{doc_id}.txt", encoding="utf-8") as f:
        return f.read()


def build_collocations(text: str, stop_words: Set[str]):
    text = text.lower()
    words = tuple(
        filter(
            lambda x: x not in stop_words,
            map(lambda x: x.group(), WORD_PATTERN.finditer(text)),
        )
    )

    # build unigram index
    unigrams = collections.defaultdict(int)
    for w in words:
        unigrams[w] += 1

    # normalize unigrams
    for ug in unigrams:
        unigrams[ug] /= len(unigrams)

    # build bigram index
    bigrams = collections.defaultdict(int)
    for word, collocant in itertools.pairwise(words):
        bigrams[(word, collocant)] += 1

    # normalize bigrams
    for bg in bigrams:
        bigrams[bg] /= len(bigrams)

    def bigram_mi(pair):
        x, y = pair
        return bigrams[pair] / unigrams[x] * unigrams[y]

    # filter bigram index by mutual information
    mi_index = [(p, bigram_mi(p)) for p in bigrams]
    mi_index.sort(key=lambda x: x[-1])

    cutoff = len(mi_index) // 10
    bigram_collocations = {k: v for k, v in mi_index[-cutoff:]}

    return unigrams, bigram_collocations


@dataclass(frozen=True, slots=True)
class DocPlainIndex:
    doc_id: str
    unigrams: Mapping[str, float]
    bigrams: Mapping[Tuple[str, str], float]

    def has_word(self, word: str):
        return word in self.unigrams

    def has_bigram(self, bigram: Tuple[str, str]):
        return bigram in self.bigrams
