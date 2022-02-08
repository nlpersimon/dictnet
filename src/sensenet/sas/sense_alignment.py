from collections import Counter, defaultdict, deque
from ..schema.senset_file import Senset


def create_senset_id(sense):
    word, pos, _, num = sense.sense_id.rsplit('.')
    senset_id = f'{word}.{pos}.{num}'
    return senset_id


class SenseAlignment:
    SOURCE_LIMIT = 1

    def __init__(self, word, pos, senses, sense_embeds):
        self._senses = deque(senses)
        self._sensets = None
        self._sense_embeds = sense_embeds
        self._word = word
        self._pos = pos

    def sensets(self):
        if self._sensets is None:
            self._sensets = []
            for senset_id, source_to_senses in self._group_senses_to_sensets().items():
                senset_senses = [
                    sense for senses in source_to_senses.values() for sense in senses]
                self._sensets.append(Senset(
                    senset_id=senset_id, word=self._word, pos_norm=self._pos, senses=senset_senses))
        return self._sensets

    def _group_senses_to_sensets(self):
        sensets = self._initialize_sensets()
        while self._senses:
            senset_candidates = self._distribute_senses(sensets)
            senset_id, sense, _ = senset_candidates.pop()
            sensets[senset_id][sense.source].append(sense)
            while senset_candidates:
                _, sense, _ = senset_candidates.pop()
                self._senses.append(sense)
        return sensets

    def _initialize_sensets(self):
        source_count = Counter(sense.source for sense in self._senses)
        base_source, _ = source_count.most_common()[0]
        sensets = {}
        for _ in range(len(self._senses)):
            sense = self._senses.popleft()
            if sense.source != base_source:
                self._senses.append(sense)
            else:
                senset_id = create_senset_id(sense)
                sensets[senset_id] = {src: [] for src in source_count}
                sensets[senset_id][sense.source].append(sense)
        return sensets

    def _distribute_senses(self, sensets):
        senset_candidates = []
        while self._senses:
            sense = self._senses.pop()
            senset_id, similarity = self._find_most_similar_senset(
                sense, sensets)
            senset_candidates.append((senset_id, sense, similarity))
        senset_candidates.sort(key=lambda x: x[2])
        return senset_candidates

    def _find_most_similar_senset(self, sense, sensets):
        possible_sensets = []
        for senset_id, source_to_senses in sensets.items():
            if len(source_to_senses[sense.source]) < self.SOURCE_LIMIT:
                similarity = self._compute_similarity(sense, source_to_senses)
                possible_sensets.append((senset_id, similarity))
        possible_sensets.sort(key=lambda x: x[1])
        return possible_sensets[-1]

    def _compute_similarity(self, sense, source_to_senses):
        similarity = -float('inf')
        for senses in source_to_senses.values():
            for _sense in senses:
                similarity = max(similarity, self._sense_embeds.similarity(
                    sense.sense_id, _sense.sense_id))
        return similarity
