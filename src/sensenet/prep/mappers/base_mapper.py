from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable
from ...schema.sense_file import SenseFileLine, SenseFileFields


class BaseMapper(ABC):
    def __init__(self,
                 sense_id: str = 'sense_id',
                 word: str = 'word',
                 pos: str = 'pos',
                 source: str = 'source',
                 definition: str = 'definition') -> None:
        self._fields = SenseFileFields(
            sense_id=sense_id,
            word=word,
            pos=pos,
            source=source,
            definition=definition)
        self._id_number = defaultdict(lambda: defaultdict(lambda: 1))

    @property
    def fields(self):
        return self._fields

    def read(self, file_pointer) -> Iterable[SenseFileLine]:
        for line in self._read(file_pointer):
            yield self.to_sense_file_line(line)

    @abstractmethod
    def _read(self, file_pointer):
        pass

    @abstractmethod
    def to_sense_file_line(self, file_line) -> SenseFileLine:
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
