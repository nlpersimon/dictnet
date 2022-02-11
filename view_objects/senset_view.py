from src.sensenet.sensenet.senset import Senset
from .sense_view import SenseView


class SensetView:
    def __init__(self,
                 senset: Senset,
                 level: str = '',
                 guideword: str = '',
                 camb_ch_def: str = ''):
        self.senset_id = senset.senset_id
        self.word = senset.word
        self.pos_norm = senset.pos_norm
        self.level = level
        self.guideword = guideword
        self.senses = [SenseView(sense, camb_ch_def)
                       for sense in senset.senses]

    def to_json(self):
        return {
            'senset_id': self.senset_id,
            'word': self.word,
            'pos_norm': self.pos_norm,
            'level': self.level,
            'guideword': self.guideword,
            'senses': [sense.to_json() for sense in self.senses]
        }
