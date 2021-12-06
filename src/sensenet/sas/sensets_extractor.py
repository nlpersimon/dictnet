from copy import deepcopy
import tqdm
from .sense_alignment import SenseAlignment


class SensetsExtractor:
    def __init__(self, senses, sense_embeds) -> None:
        self._senses = senses
        self._sense_embeds = sense_embeds
        self._senset_table = self._initialize_senset_table()

    def _initialize_senset_table(self):
        word_pos_set = set((sense.word, sense.pos_norm)
                           for sense in self._senses)
        senset_table = {wp_pair: None for wp_pair in word_pos_set}
        return senset_table

    def sensets(self, word, pos):
        wp_pair = (word, pos)
        assert wp_pair in self._senset_table, f'({word}, {pos}) is not in the collection of sensets.'
        if self._senset_table[wp_pair] is None:
            restricted_senses = self.select_senses(word, pos)
            sensets = SenseAlignment(
                word, pos, restricted_senses, self._sense_embeds).sensets()
            self._senset_table[wp_pair] = sensets
        else:
            sensets = self._senset_table[wp_pair]
        return sensets

    def all_sensets(self):
        for (word, pos), sensets in tqdm.tqdm(self._senset_table.items()):
            if sensets is None:
                self.sensets(word, pos)
        return deepcopy(self._senset_table)

    def select_senses(self, word, pos):
        return [sense for sense in self._senses
                if sense.word == word and sense.pos_norm == pos]
