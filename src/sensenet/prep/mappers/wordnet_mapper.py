from nltk.corpus import wordnet as wn
from .base_mapper import BaseMapper
from ...schema.sense_file import SenseFileLine


class WordnetMapper(BaseMapper):
    SOURCE_NAME = 'wordnet'
    SOURCE_ABBREV = 'wn'

    def __init__(self,
                 sense_id: str = 'sense_id',
                 word: str = 'word',
                 pos: str = 'pos',
                 source: str = 'source',
                 definition: str = 'definition') -> None:
        super().__init__(
            sense_id=sense_id,
            word=word,
            pos=pos,
            source=source,
            definition=definition)

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
