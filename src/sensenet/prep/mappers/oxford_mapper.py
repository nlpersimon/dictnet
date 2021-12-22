import json
from .base_mapper import BaseMapper


class OxfordMapper(BaseMapper):
    SOURCE_NAME = 'oxford'
    SOURCE_ABBREV = 'oxf'
    OXF_TO_UNI = {
        'noun': 'NOUN',
        'adjective': 'ADJ',
        'transitive verb': 'VERB',
        'verb': 'VERB',
        'intransitive verb': 'VERB',
        'adverb': 'ADV',
        'abbreviation': 'X',
        'proper noun': 'PROPN',
        'plural noun': 'NOUN',
        'exclamation': 'NOUN',
        '': 'X',
        'preposition': 'X',
        'pronoun': 'PRON',
        'suffix': 'X',
        'conjunction': 'CCONJ',
        'cardinal number': 'NUM',
        'prefix': 'X',
        'determiner': 'DET',
        'ordinal number': 'NUM',
        'adjective & adverb': 'X',
        'possessive determiner': 'DET',
        'mass noun': 'NOUN',
        'modal verb': 'X',
        'relative adverb': 'ADV',
        'auxiliary verb': 'AUX',
        'contraction': 'X',
        'noun & adjective': 'X',
        'possessive pronoun': 'PRON',
        'determiner & pronoun': 'X',
        'predeterminer, determiner, & pronoun': 'X',
        'predeterminer': 'DET',
        'relative pronoun & determiner': 'X',
        'phrase': 'X',
        'determiner & adjective': 'X',
        'pronoun & determiner': 'X',
        'conjunction & adverb': 'X',
        'determiner, predeterminer, & pronoun': 'X',
        'infinitive particle': 'ADP',
        'relative adverb & conjunction': 'X',
        'adverb & conjunction': 'X',
        'interrogative pronoun & determiner': 'X',
        'relative pronoun': 'PRON',
        'possessive determiner & pronoun': 'X',
        'pronoun, determiner, & interrogative adverb': 'X',
        'predeterminer, pronoun, & adjective': 'X'
    }

    def __init__(self) -> None:
        super().__init__()

    # def _read(self, file_pointer):
    #     oxford = json.load(file_pointer)
    #     for headword, pos_senses in oxford.items():
    #         for pos, senses in pos_senses.items():
    #             for sense in senses:
    #                 sense_data = {'headword': headword, 'pos': pos}
    #                 sub_senses = sense['subSenses']
    #                 if not sub_senses:
    #                     yield {**sense_data, 'en_def': sense['mainSense']['Def']}
    #                 else:
    #                     for sub_sense in sub_senses:
    #                         yield {**sense_data, 'en_def': sub_sense['Def']}

    def _read(self, file_pointer):
        oxford = json.load(file_pointer)
        for headword, pos_senses in oxford.items():
            for pos, senses in pos_senses.items():
                sense_data = {'headword': headword, 'pos': pos}
                for sense in senses:
                    yield {**sense_data, 'en_def': sense['mainSense']['Def']}
                    for sub_sense in sense['subSenses']:
                        yield {**sense_data, 'en_def': sub_sense['Def']}

    def get_word(self, raw_file_line) -> str:
        return raw_file_line['headword']

    def get_pos(self, raw_file_line) -> str:
        return raw_file_line['pos']

    def get_level(self, raw_file_line) -> str:
        return ''

    def get_definition(self, raw_file_line) -> str:
        return raw_file_line['en_def']

    def get_source_abbrev(self) -> str:
        return self.SOURCE_ABBREV

    def get_source(self) -> str:
        return self.SOURCE_NAME

    def normalize_pos(self, pos: str) -> str:
        return self.OXF_TO_UNI[pos]
