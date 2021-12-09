import os
from .base_sensenet import BaseSenseNet
from .max_sensenet import MaxSenseNet
from .mean_sensenet import MeanSenseNet


TYPE_TO_SENSE_EMBEDS = {
    'max': 'sense_embeddings.txt',
    'mean': 'senset_embeddings.txt'
}

TYPE_TO_SENSENET_CLS = {
    'max': MaxSenseNet,
    'mean': MeanSenseNet
}

def load_sensenet(data_folder_path: str, sensenet_type: str = 'max') -> BaseSenseNet:
    def join_path(file_name):
        return os.path.join(data_folder_path, file_name)

    assert sensenet_type in ('max', 'mean'), 'Please pass either "max" or "mean".'
    
    sense_embeds_file = TYPE_TO_SENSE_EMBEDS[sensenet_type]
    sensenet_cls = TYPE_TO_SENSENET_CLS[sensenet_type]
    
    senset_file_path = join_path('senset_file.jsonl')
    sense_embeds_path = join_path(sense_embeds_file)
    sense_embedder_path = join_path('sense_embedder/model.tar.gz')
    sensenet = sensenet_cls.from_path(
        senset_file_path, sense_embeds_path, sense_embedder_path)
    return sensenet
