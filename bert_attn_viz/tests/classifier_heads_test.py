from __future__ import division
import pytest
import bert_attn_viz.classifier_heads as clf
import tensorflow as tf

tf.enable_eager_execution()


@pytest.mark.parametrize('test_sequence_output, test_input_mask, test_max_seq_length, expected_output', [
    ([[[7., -9, -5],
       [4, 0, -1],
       [-7, 4, -8],
       [-8, 1, 6],
       [1, 7, -10]],
      [[-8., 4, -9],
       [-2, 4, 5],
       [7, 9, -9],
       [5, -2, -1],
       [20, 8, 2]]],
     [[1, 1, 1, 0, 0],
      [1, 1, 1, 1, 0]],
     5,
     [[[7, 4, -1]],
      [[7, 9, 5]]]

     )
])
def test_global_max_pool_sequence(test_sequence_output, test_input_mask, test_max_seq_length, expected_output):
    test_sequence_output = tf.constant(test_sequence_output)
    test_input_mask = tf.constant(test_input_mask, tf.int32)
    expected_output = tf.constant(expected_output, tf.float32)

    actual_output = clf._global_max_pool_sequence(test_sequence_output,
                                                  test_input_mask,
                                                  test_max_seq_length)

    assert tf.reduce_all(tf.equal(actual_output, expected_output))


@pytest.mark.parametrize('test_sequence_output, test_input_mask, test_max_seq_length, expected_output', [
    ([[[7., -9, -5],
       [4, 0, -1],
       [-7, 4, -8],
       [-8, 1, 6],
       [1, 7, -10]],
      [[-8., 4, -9],
       [-2, 4, 5],
       [7, 9, -9],
       [5, -2, -1],
       [20, 8, 2]]],
     [[1, 1, 1, 0, 0],
      [1, 1, 1, 1, 0]],
     5,
     [[[4 / 3, -5 / 3, -14 / 3]],
      [[2 / 4, 15 / 4, -14 / 4]]]

     )
])
def test_global_avg_pool_sequence(test_sequence_output, test_input_mask, test_max_seq_length, expected_output):
    test_sequence_output = tf.constant(test_sequence_output)
    test_input_mask = tf.constant(test_input_mask, tf.int32)
    expected_output = tf.constant(expected_output, tf.float32)

    actual_output = clf._global_avg_pool_sequence(test_sequence_output,
                                                  test_input_mask,
                                                  test_max_seq_length)

    assert tf.reduce_all(tf.equal(actual_output, expected_output))
