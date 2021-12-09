import os
from .base_sensenet import BaseSenseNet
from .max_sensenet import MaxSenseNet


def load_sensenet(data_folder_path: str) -> BaseSenseNet:
    def join_path(file_name):
        return os.path.join(data_folder_path, file_name)

    senset_file_path = join_path('senset_file.jsonl')
    sense_embeds_path = join_path('sense_embeddings.txt')
    sense_embedder_path = join_path('sense_embedder/model.tar.gz')
    sensenet = MaxSenseNet.from_path(
        senset_file_path, sense_embeds_path, sense_embedder_path)
    return sensenet
