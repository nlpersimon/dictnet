from nltk.corpus import wordnet as wn
from .base_mapper import BaseMapper
from ...schema.sense_file import SenseFileLine


class WordnetMapper(BaseMapper):
    SOURCE_NAME = 'wordnet'

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

    def to_sense_file_line(self, file_line) -> SenseFileLine:
        word = file_line['word']
        synset = file_line['synset']
        sense_file_line = SenseFileLine(
            sense_id=self.get_sense_id(word, synset.pos(), 'wn'),
            word=word,
            pos=synset.pos(),
            source=self.SOURCE_NAME,
            definition=synset.definition())
        return sense_file_line
