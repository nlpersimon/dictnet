from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
from typing import List
from overrides import overrides



@Predictor.register('cpae_embedder')
class CpaeEmbedder(Predictor):
    """ Runtime predictor of CPAE

    It can be used to convert a definition into definition embeddings.

    * Note: It should be instantiated by passing the path to the model archive (model.tar.gz)
    * generated at training time to from_path()
    * for example: CpaeEmbedder.from_path('path/to/model.tar.gz')
    """
    @overrides
    def predict_json(self, inputs: JsonDict) -> str:
        def_embeds = self.embed_inputs(inputs)
        return '{} {}'.format(
            inputs['word'],
            ' '.join(str(val) for val in def_embeds)
        )
    
    def embed_inputs(self, inputs: JsonDict):
        """
        Args:
            inputs: a dictionary containing two keys
                (1) word (optional)
                (2) definition: need to be tokenized
        
        Returns:
            def_embeds: definition embeddings, a list consists of 300 floating points
        """
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        def_embeds = output_dict['def_embeds']
        return def_embeds

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[str]:
        instances = self._batch_json_to_instances(inputs)
        output_dicts = self.predict_batch_instance(instances)

        results = []
        for inp, od in zip(inputs, output_dicts):
            results.append('{} {}'.format(
                inp['word'],
                ' '.join(str(val) for val in od['def_embeds'])
            ))
        return results

    @overrides
    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))

        return instances

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        definition = json_dict['definition']
        # word = json_dict['word']
        return self._dataset_reader.sense_to_instance(definition=definition, word=None)
    
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return outputs + '\n'
