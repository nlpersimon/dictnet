from .cpae.predictors import CpaeEmbedder


def load_embedder(model_path):
    embedder = CpaeEmbedder.from_path(model_path, 'cpae_embedder')
    return embedder
