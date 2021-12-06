local pretrained_embedding = "/home/nlplab/simon/python_projects/cpae-legacy/data/GoogleNews-vectors-negative300.txt";
local output_vocab_size = 17546;
local word_namespace = "tokens";
local output_namespace = "output";

{
    "dataset_reader" : {
        "type": "sense_file",
        "tokenizer": "whitespace",
        "input_token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": word_namespace,
            },
        },
        "word_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": word_namespace,
            },
        },
        "output_token_indexers": {
            "output": {
                "type": "single_id",
                "namespace": output_namespace
            },
        },
        "max_len": 100,
    },
    "train_data_path": "../../../data/v0.0.1/sense_file_bi-camb.jsonl",
    "vocabulary": {
        "max_vocab_size": {
            [word_namespace]: 50000,
        },
        "min_count": {
            [output_namespace]: 5,
        },
        "pretrained_files": {
            [word_namespace]: pretrained_embedding
        },
        "only_include_pretrained_words": true,
    },
    "model": {
        "type": "cpae",
        "word_namespace": word_namespace,
        "output_namespace": output_namespace,
        "def_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": pretrained_embedding,
                    "trainable": false,
                    "vocab_namespace": word_namespace
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
        },
        "alpha": 1.0,
        "lambda_": 64.0
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0003,
        },
        "num_epochs": 50,
        "grad_clipping": 5.0,
        "cuda_device": 1,
    }
}