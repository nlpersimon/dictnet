# Introduction
This repo is to reproduce [CPAE](https://www.aclweb.org/anthology/D18-1181/) with a fancy deep learning framework, [AllenNLP](https://github.com/allenai/allennlp), for research usage.

In short, CPAE is a deep learning model to transform a definition into a vector that can reflect its semantic characteristics.

AllenNLP is a PyTorch framework for NLP. It abstracts the process of NLP tasks into several components so that we can reduce a lot of duplicated and messy works, such as the training loop, gradient clipping, or passing data among components. If you are not familiar with AllenNLP, please refer to their fantastic [tutorial](https://guide.allennlp.org/).

# Setup
> The python version used is **3.8**

First, create a new virtual environment (recommended) and install the dependencies
```bash
pip install -r requirements.txt
```

Second, download the [data](https://drive.google.com/drive/folders/1U8NljkuBTUd-sjxdamS-IJYQ2IOJU1yt?usp=sharing) for modeling.

Once we have done, the structure of the project should look like
```bash
├── cpae 
├── model_config
├── model
│   └── separate_sense_cpae_en-camb_unfreeze-w2v
├── embeddings
│   └── separate_sense_cpae_en-camb_unfreeze-w2v.txt
├── data 
│   ├── benchmark 
│   ├── en_wn_full 
│   └── cambridge
├── requirements.txt   
├── README.md 
└── .gitignore 
```

# Training
We can train a model using allennlp train command with a configuration file easily.
```bash
allennlp train model_config/separate_sense_cpae_en-camb_unfreeze-w2v.jsonnet -s model/separate_sense_cpae_en-camb_unfreeze-w2v -f --include-package cpae
```
> Note: The existing model/separate_sense_cpae_en-camb_unfreeze-w2v will be replaced if you run the above command. You can change the `-s` argument to keep the original model.

# Prediction
## A Dictionary
Once we have got a model, we can derive definition embeddings of a dictionary, and the definition embeddings would be output as the word2vec format that `gensim.models.KeyedVectors` can read.
```bash
chmod 755 ./embed.sh

./embed.sh model/separate_sense_cpae_en-camb_unfreeze-w2v/model.tar.gz data/cambridge/processed.cambridge.sense.000.jsonl embeddings/separate_sense_cpae_en-camb_unfreeze-w2v.txt
```
> Note: The existing embeddings/separate_sense_cpae_en-camb_unfreeze-w2v.txt will be replaced if you run the above command. You can change the last argument to keep the original embeddings.
## A Single Definition
Also, we can use the model in programs to transform a definition into definition embeddings.
```python
from nltk import word_tokenize
from gensim.models import KeyedVectors
import numpy as np
from cpae.predictors import CpaeEmbedder

def_embeds = KeyedVectors.load_word2vec_format(
    'embeddings/separate_sense_cpae_en-camb_unfreeze-w2v.txt',
    binary=False
)

cpae = CpaeEmbedder.from_path(
    'model/separate_sense_cpae_en-camb_unfreeze-w2v/model.tar.gz',
    'cpae_embedder',
)

# we need to tokenize the defintion before we transform it.
definition = ' '.join(word_tokenize('a round fruit with firm, white flesh and a green, red, or yellow skin'))

inputs = {
    'word': None,
    'definition': definition
}

apple_embeds = cpae.embed_inputs(inputs)

print(def_embeds.similar_by_vector(np.array(apple_embeds)))
```