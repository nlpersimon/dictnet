from nltk.corpus import wordnet as wn
from .base_mapper import BaseMapper


class WordnetMapper(BaseMapper):
    SOURCE_NAME = 'wordnet'
    SOURCE_ABBREV = 'wn'
    WN_TO_UNI = {
        'a': 'ADJ',
        'v': 'VERB',
        'n': 'NOUN',
        'r': 'ADV',
        's': 'ADJ'
    }

    def __init__(self) -> None:
        super().__init__()

    def _read(self, file_pointer=None):
        for lemma in wn.all_lemma_names():
            for synset in wn.synsets(lemma):
                yield {'word': lemma, 'synset': synset}

    def get_word(self, raw_file_line) -> str:
        return raw_file_line['word']

    def get_source_abbrev(self) -> str:
        return self.SOURCE_ABBREV

    def get_source(self) -> str:
        return self.SOURCE_NAME

    def get_pos(self, raw_file_line) -> str:
        synset = raw_file_line['synset']
        return synset.pos()

    def get_definition(self, raw_file_line) -> str:
        synset = raw_file_line['synset']
        return synset.definition()

    def normalize_pos(self, pos: str) -> str:
        return self.WN_TO_UNI[pos]
