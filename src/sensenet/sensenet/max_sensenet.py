from gensim.models import KeyedVectors
from typing import List, Dict
from nltk.tokenize import word_tokenize
import numpy as np
from .senset import Senset
from .base_sensenet import BaseSenseNet
from ..schema.sense_file import Sense


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


class MaxSenseNet(BaseSenseNet):
    def __init__(self,
                 sensets: List[Senset],
                 sense_embeds: KeyedVectors,
                 sense_embedder) -> None:
        super().__init__(sensets, sense_embeds, sense_embedder)

    def find_similar_sensets_by_id(self, senset_id: str,
                                   pos_norm: str = None, topn: int = 10) -> List[Senset]:
        senset = self.senset(senset_id)
        base_sense = _pick_base_sense(senset.senses)
        similar_sensets = self._find_similar_sensets_by_sense_id(
            base_sense.sense_id, pos_norm, topn)
        return similar_sensets

    def reverse_dictionary(self,
                           definition: str,
                           pos_norm: str = None,
                           topn: int = 10) -> List[Senset]:
        definition = ' '.join(word_tokenize(definition))
        similar_sensets = self._find_similar_sensets_by_definition(
            definition, pos_norm, topn)
        return similar_sensets

    def _find_similar_sensets_by_definition(self,
                                            definition: str,
                                            pos_norm: str = None,
                                            topn: int = 10) -> List[Senset]:
        sensets_similarity = self._sensets_similarity_by_definition(definition)
        similar_sensets = self._filter_similar_sensets(
            sensets_similarity, pos_norm, topn)
        return similar_sensets

    def _sensets_similarity_by_definition(self, definition: str) -> Dict[str, float]:
        def_embeds = np.array(self._sense_embedder.embed_inputs({
            'definition': definition
        }))
        sensets_similarity = {}
        for sense_id, similarity in self._sense_embeds.similar_by_vector(
                def_embeds, topn=len(self._sense_embeds.vocab)):
            similar_senset_id = self._sense_to_senset[sense_id]
            sensets_similarity.setdefault(similar_senset_id, similarity)
        return sensets_similarity

    def _find_similar_sensets_by_sense_id(self,
                                          sense_id: str,
                                          pos_norm: str = None,
                                          topn: int = 10) -> List[Senset]:
        sensets_similarity = self._sensets_similarity_by_sense_id(sense_id)
        similar_sensets = self._filter_similar_sensets(
            sensets_similarity, pos_norm, topn)
        return similar_sensets

    def _sensets_similarity_by_sense_id(self, sense_id: str) -> Dict[str, float]:
        sensets_similarity = {}
        for sense_id, similarity in self._sense_embeds.most_similar(
                sense_id, topn=len(self._sense_embeds.vocab)):
            similar_senset_id = self._sense_to_senset[sense_id]
            sensets_similarity.setdefault(similar_senset_id, similarity)
        return sensets_similarity

    def _filter_similar_sensets(self,
                                sensets_similarity,
                                pos_norm: str = None,
                                topn: int = 10) -> List[Senset]:
        similar_sensets = []
        for similar_senset_id, similarity in sorted(
                sensets_similarity.items(),
                key=lambda x: x[1], reverse=True):
            _, _pos_norm, _ = similar_senset_id.rsplit('.', 2)
            if pos_norm is None or _pos_norm == pos_norm:
                similar_sensets.append(
                    (self.senset(similar_senset_id), similarity))
        return similar_sensets[:topn]
