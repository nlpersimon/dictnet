from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable
from ...schema.sense_file import SenseFileLine, SenseFileFields


class BaseMapper(ABC):
    def __init__(self,
                 sense_id: str = 'sense_id',
                 word: str = 'word',
                 pos: str = 'pos',
                 pos_norm: str = 'pos_norm',
                 source: str = 'source',
                 definition: str = 'definition') -> None:
        self._fields = SenseFileFields(
            sense_id=sense_id,
            word=word,
            pos=pos,
            pos_norm=pos_norm,
            source=source,
            definition=definition)
        self._id_number = defaultdict(lambda: defaultdict(lambda: 1))

    @property
    def fields(self):
        return self._fields

    def read(self, file_pointer) -> Iterable[SenseFileLine]:
        for line in self._read(file_pointer):
            word = self.get_word(line)
            pos = self.get_pos(line)
            source_abbrev = self.get_source_abbrev()
            source = self.get_source()
            definition = self.get_definition(line)
            pos_norm = self.normalize_pos(pos)
            sense_id = self.get_sense_id(word, pos_norm, source_abbrev)
            yield SenseFileLine(
                sense_id=sense_id,
                word=word,
                pos=pos,
                pos_norm=pos_norm,
                source=source,
                definition=definition
            )

    @abstractmethod
    def _read(self, file_pointer):
        pass

    @abstractmethod
    def get_word(self, raw_file_line) -> str:
        pass

    @abstractmethod
    def get_pos(self, raw_file_line) -> str:
        pass

    @abstractmethod
    def get_source_abbrev(self) -> str:
        pass

    @abstractmethod
    def get_source(self) -> str:
        pass

    @abstractmethod
    def get_definition(self, raw_file_line) -> str:
        pass

    @abstractmethod
    def normalize_pos(self, pos: str) -> str:
        pass

    def get_sense_id(self, word, pos, source_abbrev):
        number = self._pull_number(word, pos)
        num_str = str(number).zfill(2)
        word = word.replace(' ', '_')
        sense_id = f'{word}.{pos}.{source_abbrev}.{num_str}'
        return sense_id

    def _pull_number(self, word, pos):
        number = self._id_number[word][pos]
        self._id_number[word][pos] += 1
        return number
