from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable


@dataclass
class BaseFileLine:
    pass


class _BaseFile:
    def __init__(self, file_pointer, fields) -> None:
        self._file_pointer = file_pointer
        self._fields = fields

    @property
    def file_pointer(self):
        return self._file_pointer

    @property
    def fields(self):
        return self._fields


class BaseFileReader(ABC, _BaseFile):
    def __init__(self, file_pointer, fields) -> None:
        super().__init__(file_pointer, fields)

    @abstractmethod
    def read(self) -> Iterable[BaseFileLine]:
        pass


class BaseFileWriter(ABC, _BaseFile):
    def __init__(self, file_pointer, fields) -> None:
        super().__init__(file_pointer, fields)

    @abstractmethod
    def write(self, file_line: BaseFileLine) -> None:
        pass
