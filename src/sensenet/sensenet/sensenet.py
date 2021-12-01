import jsonlines
from typing import List, Dict
from .senset import Senset
from ..schema.senset_file import SensetFileReader


class SenseNet:
    def __init__(self, sensets: List[Senset]) -> None:
        self._senset_table = self._build_senset_table(sensets)
        self._id_to_senset = {senset.senset_id: senset for senset in sensets}

    def _build_senset_table(self, sensets: List[Senset]) -> Dict[str, Dict[str, List[Senset]]]:
        senset_table = {}
        for senset in sensets:
            word, pos = senset.word, senset.pos_norm
            senset_table.setdefault(word, {})
            senset_table[word].setdefault(pos, [])
            senset_table[word][pos].append(senset)
        return senset_table

    def senset(self, senset_id: str) -> Senset:
        assert senset_id in self._id_to_senset, f'{senset_id} is not in the sensets'
        return self._id_to_senset[senset_id]

    def sensets(self, word: str, pos: str = None) -> Senset:
        assert word in self._senset_table, f'{word} is not in the sensets.'
        word_sensets = self._senset_table[word]
        if pos is None:
            return [senset for pos_sensets in word_sensets.values()
                    for senset in pos_sensets]
        else:
            assert pos in word_sensets, f'{pos} is not in {list(word_sensets.keys())}'
            return word_sensets[pos]

    def all_sensets(self, pos: str = None) -> List[Senset]:
        return [senset for pos_sensets in self._senset_table.values()
                for p, sensets in pos_sensets.items()
                for senset in sensets
                if pos is None or p == pos]

    @classmethod
    def from_path(cls, senset_file_path: str) -> "SenseNet":
        with jsonlines.open(senset_file_path) as f:
            sensets = [Senset(i.senset_id, i.word, i.pos_norm, i.senses)
                       for i in SensetFileReader(f).read()]

        return cls(sensets)
