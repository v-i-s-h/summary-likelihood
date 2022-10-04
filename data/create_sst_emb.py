"""
    Script to create embeddings for SST dataset
    Source: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
"""

import os
import pickle

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    datadir = "./SST/"

    kw_args = { 'sep': '\t', 'names': ['label', 'text'] }
    df_train = pd.read_csv(os.path.join(datadir, "train.tsv"), **kw_args)
    df_val = pd.read_csv(os.path.join(datadir, "dev.tsv"), **kw_args)
    df_test = pd.read_csv(os.path.join(datadir, "test.tsv"), **kw_args)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')


    # Create embeddings for each split
    for split in [ 'dev', 'test', 'train']:
        df = pd.read_csv(os.path.join(datadir, "{}.tsv".format(split)), **kw_args)

        print("Loaded {} split. {} samples.".format(split, df.shape[0]))
        # Tokenize sentences
        sentences = df.text.tolist()
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = sentence_embeddings.numpy()
        print("Embeddings shape: {}".format(sentence_embeddings.shape))

        # Write out tensors as numpy arrays
        outfile = os.path.join(datadir, 'emb_{}.pkl'.format(split))
        with open(outfile, 'wb') as f:
            pickle.dump({
                    "embeddings": sentence_embeddings,
                    "labels": df.label.tolist()
                }, f)
        print("Embeddings written to file: {}".format(outfile))


if __name__=="__main__":
    main()