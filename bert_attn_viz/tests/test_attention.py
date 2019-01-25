import pytest
import bert_attn_viz.explain.attention as attn
import tensorflow as tf

tf.enable_eager_execution()


@pytest.fixture
def stopwords():
    return ['[CLS]', '[SEP]', '.', "'", ',', '!']


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


@pytest.mark.parametrize('test_input, expected_output', [
    ([('[CLS]', 0.09480611),
      (u'i', 0.15816912),
      (u'like', 0.15320916),
      (u'straw', 0.035164278),
      (u'##berries', 0.0641817),
      (u'and', 0.03637559),
      (u'bit', 0.07047665),
      (u'##co', 0.06576069),
      (u'##in', 0.029836398),
      ('[SEP]', 0.29202035)],
     [('[CLS]', 0.09480611),
      (u'i', 0.15816912),
      (u'like', 0.15320916),
      (u'strawberries', 0.049672989),
      (u'and', 0.03637559),
      (u'bitcoin', 0.055357912666666655),
      ('[SEP]', 0.29202035)])

])
def test_merge_wordpiece_tokens(test_input, expected_output):
    actual_output = attn.merge_wordpiece_tokens(test_input)
    assert actual_output == expected_output


@pytest.mark.parametrize('test_input, expected_output', [
    ([('[CLS]', 0.058553744),
      (u'the', 0.032085586),
      (u'best', 0.04366709),
      (u'company', 0.07405403),
      (u'in', 0.010829168),
      (u'the', 0.015195758),
      (u'word', 0.015859632),
      (u'!', 0.010499731),
      (u'!', 0.015029891),
      (u'!', 0.023420386),
      (u'i', 0.074467964),
      (u'like', 0.109968185),
      (u'my', 0.050472897),
      (u'boss', 0.07799122),
      (u',', 0.023359247),
      (u'colleagues', 0.049394146),
      (u'and', 0.02795547),
      (u'culture', 0.04361545),
      (u'.', 0.03408243),
      (u'nothing', 0.08191314),
      (u'!', 0.022823036),
      ('[SEP]', 0.10476178)],
     [('[CLS]', 0),
      (u'the', 0.032085586),
      (u'best', 0.04366709),
      (u'company', 0.07405403),
      (u'in', 0.010829168),
      (u'the', 0.015195758),
      (u'word', 0.015859632),
      (u'!', 0),
      (u'!', 0),
      (u'!', 0),
      (u'i', 0.074467964),
      (u'like', 0.109968185),
      (u'my', 0.050472897),
      (u'boss', 0.07799122),
      (u',', 0),
      (u'colleagues', 0.049394146),
      (u'and', 0.02795547),
      (u'culture', 0.04361545),
      (u'.', 0),
      (u'nothing', 0.08191314),
      (u'!', 0),
      ('[SEP]', 0)])

])
def test_ignore_token_weights(stopwords, test_input, expected_output):
    actual_output = attn.ignore_token_weights(test_input, stopwords)
    assert actual_output == expected_output
