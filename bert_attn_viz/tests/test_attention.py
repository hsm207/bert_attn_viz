import pytest
import bert_attn_viz.explain.attention as attn
import tensorflow as tf

tf.enable_eager_execution()


@pytest.fixture
def sample_attention_weights():
    def create_dummy_weights():
        x = tf.random.normal((3, 8, 10, 10))
        x = tf.nn.softmax(x, axis=-1)

        return x

    return [
        {'layer_1': create_dummy_weights()},
        {'layer_2': create_dummy_weights()},
        {'layer_3': create_dummy_weights()}
    ]


def test_average_first_layer_by_head(sample_attention_weights):
    x = attn.average_first_layer_by_head(sample_attention_weights)
    y = attn.average_layer_i_on_token_j_by_head(0, 0, sample_attention_weights)

    assert tf.reduce_all(tf.equal(x, y))


def test_average_last_layer_by_head(sample_attention_weights):
    x = attn.average_last_layer_by_head(sample_attention_weights)
    y = attn.average_layer_i_on_token_j_by_head(-1, 0, sample_attention_weights)

    assert tf.reduce_all(tf.equal(x, y))
