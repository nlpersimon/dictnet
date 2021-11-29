import jsonlines
from dataclasses import dataclass
from typing import Iterable, List, Union, Dict
from .base_file import BaseFileReader, BaseFileWriter, BaseFileLine
from .sense_file import SenseFileLine


@dataclass
class SensetFileLine(BaseFileLine):
    senset_id: str
    word: str
    pos_norm: str
    senses: List[SenseFileLine]

    def to_json(self) -> Dict[str, Union[str, List[str, str]]]:
        return {
            'senset_id': self.senset_id,
            'word': self.word,
            'pos_norm': self.pos_norm,
            'senses': [sense.to_json() for sense in self.senses]
        }

    @classmethod
    def from_json(cls,
                  json_dict: Dict[str, Union[str, List[str, str]]]) -> "SensetFileLine":
        return cls(
            senset_id=json_dict['senset_id'],
            word=json_dict['word'],
            pos_norm=json_dict['pos_norm'],
            senses=[SenseFileLine.from_json(sense)
                    for sense in json_dict['senses']]
        )


class SensetFileReader(BaseFileReader):
    def __init__(self,
                 file_pointer) -> None:
        assert isinstance(
            file_pointer, jsonlines.Reader), "Please use jsonlines to open the file"

        super().__init__(file_pointer)

    def read(self) -> Iterable[SensetFileLine]:
        for raw_line in self.file_pointer:
            yield SensetFileLine.from_json(raw_line)


class SensetFileWriter(BaseFileWriter):
    def __init__(self,
                 file_pointer) -> None:
        assert isinstance(
            file_pointer, jsonlines.Writer), "Please use jsonlines to open the file"
        super().__init__(file_pointer)

    def write(self, file_line: SensetFileLine) -> None:
        assert isinstance(file_line, SensetFileLine)
        self.file_pointer.write(file_line.to_json())
        return
