import jsonlines
from dataclasses import dataclass
from typing import Iterable, List, Union, Dict
from .base_file import BaseFileReader, BaseFileWriter, BaseFileLine
from .sense_file import Sense


@dataclass
class Senset(BaseFileLine):
    senset_id: str
    word: str
    pos_norm: str
    senses: List[Sense]

    def to_json(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        return {
            'senset_id': self.senset_id,
            'word': self.word,
            'pos_norm': self.pos_norm,
            'senses': [sense.to_json() for sense in self.senses]
        }

    @classmethod
    def from_json(cls,
                  json_dict: Dict[str, Union[str, List[Dict[str, str]]]]) -> "Senset":
        return cls(
            senset_id=json_dict['senset_id'],
            word=json_dict['word'],
            pos_norm=json_dict['pos_norm'],
            senses=[Sense.from_json(sense)
                    for sense in json_dict['senses']]
        )


class SensetFileReader(BaseFileReader):
    def __init__(self,
                 file_pointer) -> None:
        assert isinstance(
            file_pointer, jsonlines.Reader), "Please use jsonlines to open the file"

        super().__init__(file_pointer)

    def read(self) -> Iterable[Senset]:
        for raw_line in self.file_pointer:
            yield Senset.from_json(raw_line)


class SensetFileWriter(BaseFileWriter):
    def __init__(self,
                 file_pointer) -> None:
        assert isinstance(
            file_pointer, jsonlines.Writer), "Please use jsonlines to open the file"
        super().__init__(file_pointer)

    def write(self, senset: Senset) -> None:
        assert isinstance(senset, Senset)
        self.file_pointer.write(senset.to_json())
        return
