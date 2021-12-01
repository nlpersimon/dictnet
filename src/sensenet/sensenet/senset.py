from copy import deepcopy
from typing import List
from ..schema.sense_file import Sense


class Senset:
    def __init__(self,
                 senset_id: str,
                 word: str,
                 pos_norm: str,
                 senses: List[Sense]) -> None:
        self._senset_id = senset_id
        self._word = word
        self._pos_norm = pos_norm
        self._senses = senses

    @property
    def senset_id(self) -> str:
        return self._senset_id

    @property
    def word(self) -> str:
        return self._word

    @property
    def pos_norm(self) -> str:
        return self._pos_norm

    @property
    def senses(self) -> List[Sense]:
        return deepcopy(self._senses)

    def __repr__(self) -> str:
        return f"Senset('{self.senset_id}')"
