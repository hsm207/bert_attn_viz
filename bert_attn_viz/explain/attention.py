from __future__ import division
import tensorflow as tf
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring

try:
    from bert_attn_viz.run_classifier import file_based_convert_examples_to_features, file_based_input_fn_builder
except tf.flags.DuplicateFlagError:
    pass


def _parse_input_text(input_text, tokenizer, max_seq_length):
    tokens = tokenizer.tokenize(input_text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')

    return tokens


def average_last_layer_by_head(attentions):
    """
    Computes the attention attention weights by:
        1. Take the attention weights from the last multi-head attention layer assigned
           to the CLS token
        2. Average each token across attention heads
        3. Normalize across tokens

    :param attentions: list of dictionaries of the form
        {'layer_name': (batch_size, num_multihead_attn, sequence_length, sequence_length)}
    :return: a tensor of weights
    """
    last_multihead_attn = attentions[-1].values()[0]

    # For each multihead attention, get the attention weights going into the CLS token
    cls_attn = last_multihead_attn[:, :, 0, :]

    # Average across attention heads
    cls_attn = tf.reduce_mean(cls_attn, axis=1)

    # Normalize across tokens
    total_weights = tf.reduce_sum(cls_attn, axis=-1, keepdims=True)
    norm_cls_attn = cls_attn / total_weights

    return norm_cls_attn


def average_first_layer_by_head(attentions):
    """
    Computes the attention attention weights by:
        1. Take the attention weights from the first multi-head attention layer assigned
           to the CLS token
        2. Average each token across attention heads
        3. Normalize across tokens

    :param attentions: list of dictionaries of the form
        {'layer_name': (batch_size, num_multihead_attn, sequence_length, sequence_length)}
    :return: a tensor of weights
    """
    last_multihead_attn = attentions[0].values()[0]

    # For each multihead attention, get the attention weights going into the CLS token
    cls_attn = last_multihead_attn[:, :, 0, :]

    # Average across attention heads
    cls_attn = tf.reduce_mean(cls_attn, axis=1)

    # Normalize across tokens
    total_weights = tf.reduce_sum(cls_attn, axis=-1, keepdims=True)
    norm_cls_attn = cls_attn / total_weights

    return norm_cls_attn


def average_layer_i_on_token_j_by_head(layer_index, token_index, attentions):
    """
    General function to average attention weights by heads then across tokens in layer layer_index for token_index
    :param layer_index: The layer we want to extract the attention weights from
    :param token_index: Which token of the layer we want to extract the attention e.g. [CLS], [SEP], etc
    :param attentions: list of dictionaries of the form
        {'layer_name': (batch_size, num_multihead_attn, sequence_length, sequence_length)}
    :return: a tensor of weights
    """
    target_attention_layer = attentions[layer_index].values()[0]
    token_attn = target_attention_layer[:, :, token_index, :]

    token_attn = tf.reduce_mean(token_attn, axis=1)

    total_weights = tf.reduce_sum(token_attn, axis=-1, keepdims=True)
    norm_token_attn = token_attn / total_weights

    return norm_token_attn


def viz_attention(tokens, token_weights, target_label, pred_label, pred_probs, review_id, viz_relative=False):
    """
    Returns a html string with the tokens highlighted according to its weights
    :param tokens: A list of strings
    :param token_weights: A rank 1 numpy array where each element is the weight to attach to the corresponding token
    :param target_label: The input's ground truth label
    :param pred_label: The predicted label
    :param pred_probs: The array of predicted probabilities
    :param review_id: The input's id
    :param viz_relative: boolean indicating whether to normalize token_weights by dividing it by the max weight
    :return: A html string
    """
    prob_0, prob_1 = pred_probs
    token_weights = token_weights / np.max(token_weights) if viz_relative else token_weights

    top = Element('span')
    top.set('style', 'white-space: pre-wrap;')

    title = SubElement(top, 'h3')
    title.text = review_id

    header = SubElement(top, 'p')
    header.set('style', 'white-space: pre-line')
    header.text = """
    Target label: {}
    Predicted label: {}
    Probabilities: [{:.4f}, {:.4f}]
    """.format(target_label, pred_label, prob_0, prob_1)

    input_text = SubElement(top, 'p')
    for token, weight in zip(tokens, token_weights):
        child = SubElement(input_text, 'span')
        child.set('style', 'background-color: rgba(0, 0, 256, {})'.format(weight))
        child.text = token
        child.tail = ' '

    return tostring(top)


def merge_wordpiece_tokens(paired_tokens):
    """
    Combine tokens that have been broken up during the wordpiece tokenization process
    The weights of the merged token is the average of their wordpieces
    :param paired_tokens: A list of tuples with the form (wordpiece_token, attention_weight)
    :return: A list of tuples with the form (word_token, attention_weight)
    """
    new_paired_tokens = []
    n_tokens = len(paired_tokens)

    i = 0

    while i < n_tokens:
        current_token, current_weight = paired_tokens[i]
        if current_token.startswith('##'):
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while current_token.startswith('##'):
                merged_token = merged_token + current_token.replace('##', '')
                merged_weight.append(current_weight)
                i = i + 1
                current_token, current_weight = paired_tokens[i]
            merged_weight = np.mean(merged_weight)
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1
    return new_paired_tokens


def ignore_token_weights(paired_tokens, stopwords):
    """
    Set attention weights of certain tokens to 0 given a list of stopwords
    :param paired_tokens: A list of tuple with the form (token, attention_weight)
    :param stopwords: A list of words whose attention weight we want to set to 0
    :return: paired_tokens with the some weights set to 0 based on stopwords
    """
    new_paired_tokens = []
    for token, weight in paired_tokens:
        weight = 0 if token in stopwords else weight
        new_paired_tokens.append((token, weight))

    return new_paired_tokens
