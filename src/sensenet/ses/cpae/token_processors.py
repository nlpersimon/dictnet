from os import PathLike
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.token_indexers import TokenCharactersIndexer, TokenIndexer
from allennlp.data import Vocabulary
from overrides import overrides
from typing import List, Union, Dict
import itertools



@Tokenizer.register('sense')
class SenseTokenizer(Tokenizer):
    def __init__(self,
                 sense_list_path: Union[str, PathLike],
                 max_n_sense: int) -> None:
        self.max_n_sense = max_n_sense
        with open(sense_list_path, 'r') as f:
            sense_list = [line.split('\t') for line in f.read().splitlines()]
        self.word_to_senses = {word: senses.split(' ')[:max_n_sense]
                               for word, senses in sense_list
                               if senses != ''}
    
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        cnt = self.word_to_senses.get(text, None)
        senses = self.word_to_senses.get(text, None)
        if senses is not None:
            senses = [Token(sense) for sense in senses]
        else:
            senses = [Token(text)]
        return senses


@TokenIndexer.register('token_senses')
class TokenSensesIndexer(TokenCharactersIndexer):
    def __init__(
        self,
        sense_tokenizer: SenseTokenizer,
        namespace: str = "token_senses",
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        min_padding_length: int = 0,
        token_min_padding_length: int = 0,
        use_lemma: bool = False
    ) -> None:
        super().__init__(namespace,
                         sense_tokenizer,
                         start_tokens,
                         end_tokens,
                         min_padding_length,
                         token_min_padding_length)
        self.use_lemma = use_lemma

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[List[int]]]:
        indices: List[List[int]] = []
        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            token_indices: List[int] = []
            if token.text is None:
                raise ConfigurationError(
                    "TokenCharactersIndexer needs a tokenizer that retains text"
                )
            if self.use_lemma and (token.lemma_ in self._character_tokenizer.word_to_senses):
                text = token.lemma_
            else:
                text = token.text
            for character in self._character_tokenizer.tokenize(text):
                if getattr(character, "text_id", None) is not None:
                    # `text_id` being set on the token means that we aren't using the vocab, we just
                    # use this id instead.
                    index = character.text_id
                else:
                    index = vocabulary.get_token_index(character.text, self._namespace)
                token_indices.append(index)
            indices.append(token_indices)
        return {"token_characters": indices}
