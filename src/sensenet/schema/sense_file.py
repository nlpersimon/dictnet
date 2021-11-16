import jsonlines
from dataclasses import dataclass
from typing import Iterable
from .base_file import BaseFileReader, BaseFileWriter, BaseFileLine


@dataclass
class SenseFileLine(BaseFileLine):
    sense_id: str
    word: str
    pos: str
    source: str
    definition: str


@dataclass
class SenseFileFields(SenseFileLine):
    pass


class SenseFileReader(BaseFileReader):
    def __init__(self,
                 file_pointer,
                 sense_id: str = 'sense_id',
                 word: str = 'word',
                 pos: str = 'pos',
                 source: str = 'source',
                 definition: str = 'definition') -> None:
        assert isinstance(
            file_pointer, jsonlines.Reader), "Please use jsonlines to open the file"
        fields = SenseFileFields(
            sense_id=sense_id,
            word=word,
            pos=pos,
            source=source,
            definition=definition)
        super().__init__(file_pointer, fields)

    def read(self) -> Iterable[SenseFileLine]:
        for raw_line in self.file_pointer:
            yield SenseFileLine(
                sense_id=raw_line[self.fields.sense_id],
                word=raw_line[self.fields.word],
                pos=raw_line[self.fields.pos],
                source=raw_line[self.fields.source],
                definition=raw_line[self.fields.definition]
            )


class SenseFileWriter(BaseFileWriter):
    def __init__(self,
                 file_pointer,
                 sense_id: str = 'sense_id',
                 word: str = 'word',
                 pos: str = 'pos',
                 source: str = 'source',
                 definition: str = 'definition') -> None:
        assert isinstance(
            file_pointer, jsonlines.Writer), "Please use jsonlines to open the file"
        fields = SenseFileFields(
            sense_id=sense_id,
            word=word,
            pos=pos,
            source=source,
            definition=definition)
        super().__init__(file_pointer, fields)

    def write(self, file_line: SenseFileLine) -> None:
        assert isinstance(file_line, SenseFileLine)
        self.file_pointer.write({
            self.fields.sense_id: file_line.sense_id,
            self.fields.word: file_line.word,
            self.fields.pos: file_line.pos,
            self.fields.source: file_line.source,
            self.fields.definition: file_line.definition
        })
        return
