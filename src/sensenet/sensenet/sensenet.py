from gensim.models import KeyedVectors
import jsonlines
from typing import List, Dict
from .senset import Senset
from ..schema.sense_file import Sense
from ..schema.senset_file import SensetFileReader


def _pick_base_sense(senses: List[Sense]) -> Sense:
    source_to_senses = _group_senses_by_source(senses)
    if 'cambridge' in source_to_senses:
        return source_to_senses['cambridge'][0]
    return source_to_senses['wordnet'][0]


def _group_senses_by_source(senses: List[Sense]) -> Dict[str, List[Sense]]:
    source_to_senses = {}
    for sense in senses:
        source = sense.source
        source_to_senses.setdefault(source, [])
        source_to_senses[source].append(sense)
    return source_to_senses


class SenseNet:
    def __init__(self, sensets: List[Senset], sense_embeds: KeyedVectors) -> None:
        self._senset_table = self._build_senset_table(sensets)
        self._sense_to_senset = {sense.sense_id: senset.senset_id for senset in sensets
                                 for sense in senset.senses}
        self._id_to_senset = {senset.senset_id: senset for senset in sensets}
        self._sense_embeds = sense_embeds

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
    def from_path(cls, senset_file_path: str, sense_embeds_path: str) -> "SenseNet":
        with jsonlines.open(senset_file_path) as f:
            sensets = [Senset(i.senset_id, i.word, i.pos_norm, i.senses)
                       for i in SensetFileReader(f).read()]
        sense_embeds = KeyedVectors.load_word2vec_format(
            sense_embeds_path, binary=False)
        sensenet = cls(sensets, sense_embeds)
        for senset in sensets:
            senset.register_sensenet(sensenet)
        return sensenet

    def find_similar_sensets_by_id(self, senset_id: str,
                                   pos_norm: str = None, topn: int = 10) -> List[Senset]:
        senset = self.senset(senset_id)
        base_sense = _pick_base_sense(senset.senses)
        similar_sensets = []
        for similar_senset_id, similarity in sorted(
                self._similar_senset_ids(base_sense.sense_id).items(),
                key=lambda x: x[1], reverse=True):
            _, _pos_norm, _ = similar_senset_id.rsplit('.', 2)
            if pos_norm is None or _pos_norm == pos_norm:
                similar_sensets.append(
                    (self.senset(similar_senset_id), similarity))
        return similar_sensets[:topn]

    def _similar_senset_ids(self, sense_id: str) -> Dict[str, float]:
        similar_senset_ids = {}
        for sense_id, similarity in self._sense_embeds.most_similar(sense_id, topn=len(self._sense_embeds.vocab)):
            similar_senset_id = self._sense_to_senset[sense_id]
            similar_senset_ids.setdefault(similar_senset_id, similarity)
        return similar_senset_ids
