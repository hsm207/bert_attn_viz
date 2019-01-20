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
 
# How it works
The forward pass has been modified to return a list of `{layer_i: layer_i_attention_weights}` dictionaries. The shape of 
 `layer_i_attention_weights` is `(batch_size, num_multihead_attn, max_seq_length, max_seq_length)`.
 
 You can specify a function to process the above list by passing it as a parameter into the `load_bert_model` function
 in the [explain.model](explain/model.py) module. The function's output is avaialble as part of the result of the Estimator's predict call
 under the key named 'attention'.
 
 Currently, only two attention processor functions have been defined, namely `average_last_layer_by_head` and  `average_first_layer_by_head`.
 See [explain.attention](explain/attention.py) for implementation details.       
# Model Performance Metrics

The fine-tuned model achieved an accuracy of 0.9407 on the test set. 

The fine-tuning process was done with the following hyperparameters:

* maximum sequence length: 512
* training batch size: 8
* learning rate: 3e-5
* number of epochs: 3
