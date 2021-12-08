from .base_sensenet import BaseSenseNet
from .max_sensenet import MaxSenseNet


def load_sensenet(senset_file_path: str, sense_embeds_path: str, sense_embedder_path: str) -> BaseSenseNet:
    sensenet = MaxSenseNet.from_path(
        senset_file_path, sense_embeds_path, sense_embedder_path)
    return sensenet
