from copy import deepcopy
from typing import Iterable
from nltk.corpus import stopwords
import spacy
from ..schema.sense_file import Sense

nlp = spacy.load('en_core_web_sm')


class Preprocessor:
    def __init__(self,
                 tokenize=True,
                 remove_mwe=True,
                 remove_stop_words=True) -> None:
        self._tokenize = tokenize
        self._remove_mwe = remove_mwe
        self._remove_stop_words = remove_stop_words
        self._stop_words = set(stopwords.words(
            'english')) if remove_stop_words else None

    def preprocess(self, sense_file_lines: Iterable[Sense]) -> Iterable[Sense]:
        for line in sense_file_lines:
            line = deepcopy(line)
            if self._remove_mwe and self.is_mwe(line.word):
                continue
            elif self._remove_stop_words and self.is_stop_word(line.word):
                continue
            elif self._tokenize:
                line.definition = ' '.join(self.tokenize(line.definition))
            yield line

    def is_mwe(self, word: str) -> bool:
        return ' ' in word or '_' in word or '-' in word or '.' in word

    def is_stop_word(self, word: str) -> bool:
        return word in self._stop_words

    def tokenize(self, text):
        return [tok.text for tok in nlp(text)]
