from copy import deepcopy
from typing import List, TYPE_CHECKING
from ..schema.sense_file import Sense

if TYPE_CHECKING:
    from .base_sensenet import BaseSenseNet


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
        self._sensenet = None

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

    def register_sensenet(self, sensenet: "BaseSenseNet") -> "Senset":
        self._sensenet = sensenet
        return self

    def similar_sensets(self, pos: str = None, top: int = 10) -> List["Senset"]:
        assert self._sensenet is not None, 'Please register a SenseNet first'
        return self._sensenet.find_similar_sensets_by_id(self.senset_id, pos, top)
