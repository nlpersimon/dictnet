from allennlp.training import TrainerCallback, GradientDescentTrainer
from allennlp.data import TensorDict
from allennlp.nn import util
from os import PathLike
from typing import Union, Dict, Any, List, Optional
from numpy import inner
from numpy.linalg import norm
from scipy.stats import spearmanr
import pdb



@TrainerCallback.register('similarity')
class SimilarityCallback(TrainerCallback):
    def __init__(
        self,
        serialization_dir: str,
        similarity_file: Union[str, PathLike],
        word_namespace: str
    ) -> None:
        super().__init__(serialization_dir=serialization_dir)
        self.similarity_file = similarity_file
        self.similarities = []
        self.def_embeds = {}
        self.word_namespace = word_namespace
    
    def on_start(
        self,
        trainer: GradientDescentTrainer,
        is_primary: bool = True, 
        **kwargs
    ) -> None:
        self.trainer = trainer
        if is_primary:
            with open(self.similarity_file, 'r') as f:
                for row in f.read().splitlines():
                    w1, w2, similarity = row.split('\t')
                    self.similarities.append((w1, w2, float(similarity)))
                    self.def_embeds[w1] = None
                    self.def_embeds[w2] = None
        return
    
    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs
    ) -> None:
        if not is_primary or is_training:
            return
        word_ids = util.get_token_ids_from_text_field_tensors(batch_inputs[0]['word'])
        for word_id, def_embeds in zip(word_ids.squeeze(1), batch_outputs[0]['def_embeds']):
            word = trainer._pytorch_model.vocab.get_token_from_index(int(word_id), namespace=self.word_namespace)
            if word != trainer._pytorch_model.vocab._oov_token and word in self.def_embeds:
                self.def_embeds[word] = def_embeds.detach().cpu().numpy()
        return
    
    def on_epoch(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if not is_primary:
            return
        #pdb.set_trace()
        true_sim, pred_sim = [], []
        for w1, w2, sim in self.similarities:
            w1_def_embeds = self.def_embeds.get(w1, None)
            w2_def_embeds = self.def_embeds.get(w2, None)
            if w1_def_embeds is None or w2_def_embeds is None:
                continue
            w1_def_embeds = w1_def_embeds
            w2_def_embeds = w2_def_embeds
            p_sim = inner(w1_def_embeds, w2_def_embeds) / (norm(w1_def_embeds) * norm(w2_def_embeds))
            true_sim.append(sim)
            pred_sim.append(p_sim)
        sim_spearmanr = spearmanr(true_sim, pred_sim).correlation
        metrics['validation_spearmanr'] = sim_spearmanr
        # pdb.set_trace()
        return

    def on_end(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        return
