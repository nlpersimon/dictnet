# Introduction
SenseNet is a system that:
  - aligns senses from different dictionaries but with the same meaning
  - performs sense-based reverse dictionary
  - computes similarity based on the combination of deep learning and dictionaries

`Senset` is the atomic unit of the system, and represents a set of word senses which have the same meaning. It has the following attributes:
  - senset_id (string): unique ID of a `Senset`
  - word (string): the corresponding word of a `Senset`
  - pos_norm (string): the [normalized POS tag](https://universaldependencies.org/u/pos/) of a `Senset`
  - senses (list of `Sense`): the senses whiche belong to a `Senset`  

`Sense` represents a word sense defined by a dictionary and has the following attributes:
  - sense_id (string): unique ID of a `Sense`
  - word (string): the corresponding word of a `Sense`
  - pos (string): the raw POS tag of a `Sense`
  - pos_norm (string): the [normalized POS tag](https://universaldependencies.org/u/pos/) of a `Senset`
  - source (string): the name of the source dictionary of a `Sense`
  - definition (string): the definition in English of a `Sense`

The system currently has the following characteristics:
  - Every `Sense`s of a `Senset` are from different dictionary.
  - The source of a `Sense` is either **wordnet** or **cambridge**.

# Setup
> The python version used is **3.8**

First, create a new virtual environment (recommended) and install the dependencies
```bash
pip install -r requirements.txt
```

Second, download the [data](https://drive.google.com/file/d/1kcFQ5d2re3nTi9W9t9OqGXUxC2iThGKk/view?usp=sharing) for the system.

Once we have done, the structure of the project should look like
```bash
├── scripts
├── src
├── data 
│   └── v0.0.1
│       └── wn_bi-camb
│           ├── sense_file_bi-camb.jsonl
│           ├── senset_file_bi-camb.jsonl
│           ├── wn_bi-camb_cpae.txt
│           └── wn_bi-camb_cpae
├── requirements.txt
├── README.md 
├── demo.ipynb
└── .gitignore
```

# Quick Start
Please refer to the [demo.ipynb](./demo.ipynb).