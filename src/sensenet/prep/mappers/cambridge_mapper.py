import jsonlines
from .base_mapper import BaseMapper
from ...schema.sense_file import SenseFileLine


class CambridgeMapper(BaseMapper):
    SOURCE_NAME = 'cambridge'

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

    def _read(self, file_pointer):
        assert isinstance(
            file_pointer, jsonlines.Reader), "Please use jsonlines to open the file"
        for line in file_pointer:
            yield line

    def to_sense_file_line(self, file_line) -> SenseFileLine:
        word = file_line['headword']
        pos = file_line['pos'] or 'null'
        sense_file_line = SenseFileLine(
            sense_id=self.get_sense_id(word, pos, 'camb'),
            word=word,
            pos=pos,
            source=self.SOURCE_NAME,
            definition=file_line['en_def'])
        return sense_file_line
