from gensim.models import KeyedVectors
from typing import List
from nltk.tokenize import word_tokenize
import numpy as np
from .senset import Senset
from .base_sensenet import BaseSenseNet


class MeanSenseNet(BaseSenseNet):
    def __init__(self,
                 sensets: List[Senset],
                 sense_embeds: KeyedVectors,
                 sense_embedder) -> None:
        super().__init__(sensets, sense_embeds, sense_embedder)

    def find_similar_sensets_by_id(self, senset_id: str,
                                   pos_norm: str = None, topn: int = 10) -> List[Senset]:
        similar_sensets = []
        for similar_senset_id, similarity in self._sense_embeds.most_similar(
                senset_id, topn=len(self._sense_embeds.vocab)):
            if len(similar_sensets) == topn:
                break
            senset = self.senset(similar_senset_id)
            if pos_norm is None or senset.pos_norm == pos_norm:
                similar_sensets.append((senset, similarity))
        return similar_sensets

    def reverse_dictionary(self,
                           definition: str,
                           pos_norm: str = None,
                           topn: int = 10) -> List[Senset]:
        definition = ' '.join(word_tokenize(definition))
        def_embeds = np.array(self._sense_embedder.embed_inputs({
            'definition': definition
        }))
        similar_sensets = []
        for similar_senset_id, similarity in self._sense_embeds.similar_by_vector(
                def_embeds, topn=len(self._sense_embeds.vocab)):
            if len(similar_sensets) == topn:
                break
            senset = self.senset(similar_senset_id)
            if pos_norm is None or senset.pos_norm == pos_norm:
                similar_sensets.append((senset, similarity))
        return similar_sensets
