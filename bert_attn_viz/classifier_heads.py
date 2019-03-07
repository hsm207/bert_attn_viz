import tensorflow as tf
import bert_attn_viz.modeling as modeling
import bert_attn_viz.san as san


# TODO: Finish documenting function
def custom_pooler(pooler_type,
                  sequence_outputs,
                  input_mask,
                  max_seq_length,
                  embedding_dim,
                  stdev,
                  is_training,
                  keep_dropout_rate,
                  input_ids,
                  max_relative_position):
    """
    Function to return different types of poolers to use on BERT's sequence of output.
    :param pooler_type:
    :param sequence_outputs: BERT's output which is a tensor with shape (batch size, max sequence length, hidden size)
    :param input_mask:
    :param max_seq_length:
    :param embedding_dim:
    :param stdev:
    :return:
    """
    output = None
    # TODO: check for invalid pooler_type
    with tf.variable_scope("custom_pooler"):
        if pooler_type == "global_max_pool":
            output = _global_max_pool_sequence(sequence_outputs, input_mask, max_seq_length)
        elif pooler_type == "global_average_pool":
            output = _global_avg_pool_sequence(sequence_outputs, input_mask)
        elif pooler_type == "ssan":
            output = san.single_head_attention(sequence_outputs,
                                               is_training,
                                               keep_dropout_rate,
                                               input_ids,
                                               input_mask,
                                               max_relative_position)
            output = _global_avg_pool_sequence(output, input_mask)

        output = tf.squeeze(output, axis=1)

        if pooler_type == "ssan":
            output = tf.layers.dense(output,
                                     embedding_dim,
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     kernel_initializer=tf.glorot_normal_initializer(),
                                     bias_initializer=tf.zeros_initializer())

            if is_training:
                output = tf.nn.dropout(output,
                                       keep_prob=keep_dropout_rate)
        else:

            output = tf.layers.dense(
                output,
                embedding_dim,
                activation=tf.tanh,
                kernel_initializer=modeling.create_initializer(stdev)
            )

    return output


def _global_max_pool_sequence(sequence_outputs, input_mask, max_seq_length):
    """
    Applies max pool over the embedding dimensions of the output sequence of an enocoder layer but only over non-zero
    padded sequence

    :param sequence_outputs: A tensor of shape (batch_size, max_seq_length, embedding_dim)
    :param input_mask: A binary tensor with shape (batch_size, max_seq_length) indicating which elements in the
                       input are zero-padded (0) or legit values (1)
    :param max_seq_length: The length of the sequence
    :return: A tensor with shape (batch_size, 1, embedding_dim)
    """

    # cast input mask to float so that matrix multiplication will work
    input_mask = tf.cast(tf.expand_dims(input_mask, -1), tf.float32)

    # set the zero-padded elements to 0s
    masked_sequence_outputs = sequence_outputs * input_mask

    # set the zero-padded elements to -1000 so that max pool will ignore it
    masked_sequence_outputs = masked_sequence_outputs + (1 - input_mask) * -1e3

    global_max_pool_tokens = tf.layers.max_pooling1d(masked_sequence_outputs,
                                                     pool_size=max_seq_length,
                                                     strides=max_seq_length,
                                                     padding='valid')

    return global_max_pool_tokens


def _global_avg_pool_sequence(sequence_outputs, input_mask):
    """
    Applies average pooling over the embedding dimensions of the output sequence of an enocoder layer but only
    over non-zero padded sequence

    :param sequence_outputs: A tensor of shape (batch_size, max_seq_length, embedding_dim)
    :param input_mask: A binary tensor with shape (batch_size, max_seq_length) indicating which elements in the
                       input are zero-padded (0) or legit values (1)
    :param max_seq_length: The length of the sequence
    :return: A tensor with shape (batch_size, 1, embedding_dim)
    """

    # cast input mask to float so that matrix multiplication will work
    input_mask = tf.cast(tf.expand_dims(input_mask, -1), tf.float32)

    # set the zero-padded elements to 0s
    masked_sequence_outputs = sequence_outputs * input_mask

    # sum the non-zero padded elements by the embedding_dim dimension (batch_size, embedding_dim, 1)
    masked_sequence_outputs = tf.matmul(masked_sequence_outputs, input_mask, transpose_a=True)

    # divide by the number of non zero elements in each sequence
    denoms = tf.reduce_sum(input_mask, axis=1, keep_dims=True)
    avg_pool_tokens = masked_sequence_outputs / denoms

    # reshape to (batch_size, 1, embedding_dim)
    avg_pool_tokens = tf.transpose(avg_pool_tokens, [0, 2, 1])

    return avg_pool_tokens
