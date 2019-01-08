import tensorflow as tf
import tokenization
from explain.model import load_bert_model
import tempfile
import os
from xml.etree.ElementTree import Element, SubElement, tostring

try:
    from run_classifier import CfeProcessor, \
        file_based_convert_examples_to_features, \
        file_based_input_fn_builder
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
    total_weights = tf.reduce_sum(cls_attn, axis=-1, keep_dims=True)
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
    total_weights = tf.reduce_sum(cls_attn, axis=-1, keep_dims=True)
    norm_cls_attn = cls_attn / total_weights

    return norm_cls_attn


def viz_attention(tokens, token_weights, target_label, pred_label, pred_probs, review_id):
    """
    Returns a html string with the tokens highlighted according to its weights
    :param tokens: A list of strings
    :param token_weights: A rank 1 numpy array where each element is the weight to attach to the corresponding token
    :param target_label: The input's ground truth label
    :param pred_label: The predicted label
    :param pred_probs: The array of predicted probabilities
    :param review_id: The input's id
    :return: A html string
    """
    prob_0, prob_1 = pred_probs

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


def main():
    vocab_file = './model/uncased_L-12_H-768_A-12/vocab.txt'
    do_lower_case = True
    data_dir = './data/CFE'
    output_dir = tempfile.gettempdir() if data_dir is None else data_dir
    predict_file = os.path.join(output_dir, 'predict.tf_record')
    max_seq_length = 128

    input_processor = CfeProcessor()

    label_list = input_processor.get_labels()
    estimator = load_bert_model(output_dir=output_dir,
                                num_labels=len(label_list),
                                attn_processor_fn=average_last_layer_by_head)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)

    predict_examples = input_processor.get_test_examples(data_dir)
    file_based_convert_examples_to_features(predict_examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            predict_file)
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False
    )

    predictions = estimator.predict(input_fn=predict_input_fn)
    vizs = []
    for predict_example, prediction in zip(predict_examples, predictions):
        input_tokens = _parse_input_text(predict_example.text_a, tokenizer, max_seq_length)
        input_attention = prediction['attention']
        viz = viz_attention(input_tokens,
                            input_attention,
                            predict_example.label,
                            prediction['pred_class'],
                            prediction['probabilities'],
                            predict_example.guid)

        vizs.append(viz)


if __name__ == "__main__":
    main()
