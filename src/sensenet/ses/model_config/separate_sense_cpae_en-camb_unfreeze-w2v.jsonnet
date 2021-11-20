local pretrained_embedding = "/home/nlplab/simon/python_projects/cpae-legacy/data/GoogleNews-vectors-negative300.txt";
local output_vocab_size = 17546;
local word_namespace = "tokens";
local output_namespace = "output";

{
    "dataset_reader" : {
        "type": "dictionary",
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
        "separate": true
    },
    "train_data_path": "data/cambridge/en_cambridge_all.jsonl",
    "validation_data_path": "data/benchmark/sim999_definitions.jsonl",
    "vocabulary": {
        "max_vocab_size": {
            [word_namespace]: 50000,
            [output_namespace]: output_vocab_size,
        },
        "tokens_to_add": {
            [word_namespace]: ["<sep>"],
            [output_namespace]: ["<sep>"],
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
                    "trainable": true,
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
        "callbacks": [
            {
                "type": "similarity",
                "similarity_file": "data/benchmark/sim999.txt",
                "word_namespace": word_namespace
            },
            {"type": "tensorboard"},
        ],
    }
}