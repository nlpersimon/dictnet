import jsonlines
from dataclasses import dataclass
from typing import Iterable, Dict
from .base_file import BaseFileReader, BaseFileWriter, BaseFileLine


@dataclass
class SenseFileLine(BaseFileLine):
    sense_id: str
    word: str
    pos: str
    pos_norm: str
    source: str
    definition: str

    def to_json(self) -> Dict[str, str]:
        return {
            'sense_id': self.sense_id,
            'word': self.word,
            'pos': self.pos,
            'pos_norm': self.pos_norm,
            'source': self.source,
            'definition': self.definition,
        }

    @classmethod
    def from_json(cls, json_dict: Dict[str, str]) -> "SenseFileLine":
        return cls(
            sense_id=json_dict['sense_id'],
            word=json_dict['word'],
            pos=json_dict['pos'],
            pos_norm=json_dict['pos_norm'],
            source=json_dict['source'],
            definition=json_dict['definition']
        )


class SenseFileReader(BaseFileReader):
    def __init__(self,
                 file_pointer) -> None:
        assert isinstance(
            file_pointer, jsonlines.Reader), "Please use jsonlines to open the file"

        super().__init__(file_pointer)

    def read(self) -> Iterable[SenseFileLine]:
        for raw_line in self.file_pointer:
            yield SenseFileLine.from_json(raw_line)


class SenseFileWriter(BaseFileWriter):
    def __init__(self,
                 file_pointer) -> None:
        assert isinstance(
            file_pointer, jsonlines.Writer), "Please use jsonlines to open the file"
        super().__init__(file_pointer)

    def write(self, file_line: SenseFileLine) -> None:
        assert isinstance(file_line, SenseFileLine)
        self.file_pointer.write(file_line.to_json())
        return
