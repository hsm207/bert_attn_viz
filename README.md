# Introduction

This repository is an adaptation of  the [bert](https://github.com/google-research/bert) repository.

The purpose of this repository is to visualize BERT's self-attention weights after it has been fine-tuned on the [IMDb dateset](http://ai.stanford.edu/~amaas/data/sentiment/). However, it can be extended to any text classification dataset by creating an appropriate DataProcessor class. See [run_classifier.py](run_classifier.py) for details.

# Usage

1. Create a tsv file for each of the IMDB training and test set.

    Refer to the [imdb_data](https://github.com/hsm207/imdb_data) repo for instructions.

2. Fine-tune BERT on the IMBDb training set.

   Refer to the official BERT repo for fine-tuning instructions.
    
   Alternatively, you can skip this step by downloading the fine-tuned model from [here](https://drive.google.com/open?id=13Ajyk6xejy3kRU7Ewo_5slCo9db2bOdk).
   
   The pre-trained model (BERT base uncased) used to perform the fine-tuning can also be downloaded from [here](https://drive.google.com/open?id=1f23aE84MlPY1eQqzyENt4Fk_DGucof_4). 

3. Visualize BERT's weights.
   
   Refer to the [BERT_viz_attention_imdb](/notebooks/BERT_viz_attention_imdb.ipynb) notebook for more details.
   
# Model Performance Metrics

The fine-tuned model achieved an accuracy of 0.9407 on the test set. 

The fine-tuning process was done with the following hyperparameters:

* maximum sequence length: 512
* training batch size: 8
* learning rate: 3e-5
* number of epochs: 3
