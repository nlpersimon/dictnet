from multiprocessing.sharedctypes import Value
from typing import Dict
from src.sensenet.schema.sense_file import Sense


class SenseView:
    def __init__(self, sense: Sense, camb_ch_def: str = '') -> None:
        self.sense_id = sense.sense_id
        self.word = sense.word
        self.pos = sense.pos
        self.pos_norm = sense.pos_norm
        self.source = sense.source
        self.en_def = sense.definition
        self.ch_def = camb_ch_def if self.source == 'cambridge' else ''
        self.source_url = self.construct_src_url()

    def construct_src_url(self):
        if self.source == 'cambridge':
            domain_url = 'https://dictionary.cambridge.org/dictionary/english-chinese-traditional/'
        elif self.source == 'wordnet':
            domain_url = 'http://wordnetweb.princeton.edu/perl/webwn?s='
        else:
            raise ValueError
        return domain_url + self.word

    def to_json(self) -> Dict[str, str]:
        return {
            'sense_id': self.sense_id,
            'word': self.word,
            'pos': self.pos,
            'pos_norm': self.pos_norm,
            'source': self.source,
            'en_def': self.en_def,
            'ch_def': self.ch_def,
            'source_url': self.source_url
        }
