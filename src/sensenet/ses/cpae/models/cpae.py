from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from typing import Dict
import torch
import torch.nn as nn
import pdb


@Model.register('cpae')
class CPAE(Model):
    """ Train the model

    Args:
        vocab: an allennlp Vocabulary, contains index spaces of fields of the instances
        word_namespace: namespace of word (aka headword) to index vocabulary
        output_namespace: namespace of output tokens to index vocabulary
        def_embedder: convert the input definition tokens and word tokens to word embeddings
        encoder: convert the word embeddings of an definition from multiple vectors to
                 a single vector which represents the definition, a.k.a definition embeddings
        alpha: weight to reconstruction loss
        lambda_: weight to consistency loss
        word_embedder: embed the input word (headword). if it doesn't be specified, it would be def_embedder
    """

    def __init__(self,
                 vocab: Vocabulary,
                 word_namespace: str,
                 output_namespace: str,
                 def_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 alpha: float,
                 lambda_: float,
                 word_embedder: TextFieldEmbedder = None) -> None:
        super().__init__(vocab)
        self.oov_index = self.vocab.get_token_index(
            self.vocab._oov_token, word_namespace)
        self.def_embedder = def_embedder
        self.word_embedder = word_embedder or def_embedder
        self.encoder = encoder
        self.W_def = nn.Linear(self.encoder.get_output_dim(),
                               self.encoder.get_output_dim(),
                               bias=True)
        self.classifier = nn.Linear(self.encoder.get_output_dim(),
                                    self.vocab.get_vocab_size(output_namespace))
        self.alpha = alpha
        self.lambda_ = lambda_

    def forward(self,
                sense_in: TextFieldTensors,
                word: TextFieldTensors = None,
                sense_out: TextFieldTensors = None) -> Dict[str, torch.Tensor]:
        # (batch_size, max_len, embed_dim)
        input_embeds = self.def_embedder(sense_in)
        mask = util.get_text_field_mask(sense_in)
        encoding = self.encoder(input_embeds, mask=mask)
        # (batch_size, hidden_dim)
        def_embeds = self.W_def(encoding)
        output = {'def_embeds': def_embeds}
        if word is not None:
            output_tokens = util.get_token_ids_from_text_field_tensors(
                sense_out)
            loss = self.compute_loss(def_embeds,
                                     output_tokens,
                                     mask,
                                     word)
            output['loss'] = loss
        return output

    def compute_loss(self,
                     def_embeds: torch.Tensor,
                     output_tokens: torch.Tensor,
                     mask: torch.Tensor,
                     word: TextFieldTensors) -> torch.Tensor:
        ce_loss = self.compute_ce_loss(def_embeds, output_tokens, mask)
        loss = self.alpha * ce_loss
        if self.lambda_ > 0:
            consistency_penalty = self.compute_consistency_penalty(
                def_embeds, word)
            loss += self.lambda_ * consistency_penalty
        return loss

    def compute_ce_loss(self,
                        def_embeds: torch.Tensor,
                        output_tokens: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        # (batch_size, vocab_size)
        logits = self.classifier(def_embeds)
        _, max_len = output_tokens.shape
        # (batch_size, max_len, vocab_size)
        expanded_logits = logits.unsqueeze(1).repeat(1, max_len, 1)
        ce_loss = util.sequence_cross_entropy_with_logits(expanded_logits,
                                                          output_tokens,
                                                          weights=mask,
                                                          average='token')
        return ce_loss

    def compute_consistency_penalty(self,
                                    def_embeds: torch.Tensor,
                                    word: TextFieldTensors) -> torch.Tensor:
        # (batch_size, embed_dim)
        word_embeds = self.word_embedder(word).squeeze(1)
        # (batch_size, )
        word_ids = util.get_token_ids_from_text_field_tensors(word).squeeze(1)
        weights = (word_ids != self.oov_index).float()
        mse = torch.mean((word_embeds - def_embeds)**2, dim=1)
        c_w = (mse * weights).sum() / (weights.sum() + 1e-8)
        return c_w
