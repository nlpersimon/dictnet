#!/bin/bash -e


MODEL_PATH=$1
INPUT_PATH=$2
OUTPUT_PATH=$3



# generate definition embeddings
allennlp predict "$MODEL_PATH" "$INPUT_PATH" --output-file "$OUTPUT_PATH" --include-package cpae --predictor cpae_embedder --batch-size 32 --cuda 1 --silent


# count the number of the embeddings
EMBED_COUNT=`wc -l "$OUTPUT_PATH" | awk '{print $1}'`


# insert the number of the embeddings and the dimension of that to the def-embed file
sed -i "1s/^/$EMBED_COUNT 300\n/" "$OUTPUT_PATH"
