from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField
from ..constants import SEP_TOKEN
from typing import Dict, Iterable
import jsonlines


@DatasetReader.register('sense_file')
class SenseFileReader(DatasetReader):
    """ Read jsonline files of `SenseFileLine` and convert it to allennlp instances

    Each line of a jsonline file must have three fields:
        1. word
        2. definition
        3. sense_id
        e.g., {'word': 'apple', # this is definition of apple in cambridge dictionary
               'definition': 'a round fruit with firm, white flesh and a green, red, or yellow skin',
               'sense_id': 'apple.NOUN.wn.01}


    Args:
        tokenizer: allennlp tokenizer which supports method, tokenize(),
                    can tokenize a sentence into tokens
        input_token_indexers: token indexers for input definitions
        word_indexers: token indexers for words regarding to the input definitions,
                        we usally want word_indexers and input_token_indexers are in the same index space
        output_token_indexers: token indexers for input definitions that map them
                                to the other index space used by auto-encoder
        max_len: max length of an input definition
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 input_token_indexers: Dict[str, TokenIndexer],
                 word_indexers: Dict[str, TokenIndexer] = None,
                 output_token_indexers: Dict[str, TokenIndexer] = None,
                 max_len: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.input_token_indexers = input_token_indexers
        self.output_token_indexers = output_token_indexers
        self.word_indexers = word_indexers
        self.max_len = max_len

    def _read(self, file_path: str) -> Iterable[Instance]:
        with jsonlines.open(file_path, 'r') as f:
            for row in f:
                word = row.get('word', None)
                instance = self.sense_to_instance(row['definition'], word)
                yield instance

    def sense_to_instance(self,
                          definition: str,
                          word: str = None) -> Instance:
        def_tokens = self.tokenizer.tokenize(definition)
        if self.max_len:
            def_tokens = def_tokens[:self.max_len]
        if def_tokens[-1].text == SEP_TOKEN:
            def_tokens.pop()
        fields = {
            'sense_in': TextField(def_tokens, self.input_token_indexers),
        }
        if word is not None:
            fields['word'] = TextField(
                [Token(word, lemma_=word)], self.word_indexers)
            if self.output_token_indexers is not None:
                fields['sense_out'] = TextField(
                    def_tokens, self.output_token_indexers)
        return Instance(fields)
