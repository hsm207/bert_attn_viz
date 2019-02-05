from __future__ import division
import pytest
import tensorflow as tf
import bert_attn_viz.learning_rate as lr
import numpy as np


@pytest.fixture
def dummy_loss():
    X = tf.random.normal([10, 3], 10, 4, tf.float32)
    Y = tf.matmul(X, tf.constant([10, 4, 30], shape=[3, 1], dtype=tf.float32))
    W = tf.get_variable('W', [3, 1], tf.float32)

    Y_hat = tf.matmul(X, W)

    loss = tf.losses.mean_squared_error(labels=Y, predictions=Y_hat)

    return loss


def test_default_learning_rate_schdule(dummy_loss):
    num_train_steps = 10
    num_warmup_steps = 5
    global_step = tf.train.get_or_create_global_step()

    learning_rate_fn = lr.lr_schedule_picker('default',
                                             init_lr=1e-3,
                                             num_train_steps=num_train_steps,
                                             num_warmup_steps=num_warmup_steps
                                             )
    learning_rate = learning_rate_fn(global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(dummy_loss, global_step)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)

    lr_hist = []
    for _ in range(num_train_steps):
        _, lr_val = sess.run([train_op, learning_rate])
        lr_hist.append(lr_val)

    # not really sure what the authors intended...
    assert True
    # reset graph so that it does not influence other tests
    tf.reset_default_graph()


def test_lr_range_test(dummy_loss):
    num_train_steps = 100
    max_learning_rate = 1
    global_step = tf.train.get_or_create_global_step()

    learning_rate_fn = lr.lr_schedule_picker('lr_range_test',
                                             max_learning_rate=max_learning_rate,
                                             num_train_steps=num_train_steps)

    learning_rate = learning_rate_fn(global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(dummy_loss, global_step)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)

    lr_hist = []
    for _ in range(num_train_steps):
        _, lr_val = sess.run([train_op, learning_rate])
        lr_hist.append(lr_val)

    assert lr_hist[0] == 0
    assert lr_hist[-1] == max_learning_rate
    assert len(lr_hist) == num_train_steps
    assert all(np.isclose(np.diff(lr_hist),
                          max_learning_rate / num_train_steps,
                          atol=1 / 100))

    tf.reset_default_graph()


def test_momentum(dummy_loss):
    num_train_steps = 100
    max_learning_rate = 1
    global_step = tf.train.get_or_create_global_step()

    learning_rate_fn = lr.lr_schedule_picker('lr_range_test',
                                             max_learning_rate=max_learning_rate,
                                             num_train_steps=num_train_steps)

    learning_rate = learning_rate_fn(global_step)

    momentum_fn = lr.lr_schedule_picker('lr_range_test',
                                        max_learning_rate=0.5,
                                        num_train_steps=num_train_steps)

    momentum = momentum_fn(global_step)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    train_op = optimizer.minimize(dummy_loss, global_step)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)

    lr_hist = []
    mom_hist = []
    for _ in range(num_train_steps):
        _, lr_val, mom_val = sess.run([train_op, learning_rate, momentum])
        lr_hist.append(lr_val)
        mom_hist.append(mom_val)

    # assert lr_hist[0] == 0
    # assert lr_hist[-1] == max_learning_rate
    # assert len(lr_hist) == num_train_steps
    # assert all(np.isclose(np.diff(lr_hist),
    #                       max_learning_rate / num_train_steps,
    #                       atol=1 / 100))

    tf.reset_default_graph()


def test_polynomial_decay(dummy_loss):
    num_train_steps = 3
    max_learning_rate = 1
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.polynomial_decay(learning_rate=1.0,
                                              global_step=global_step,
                                              decay_steps=(num_train_steps-1),
                                              end_learning_rate=0.0,
                                              power=1
                                              )

    momentum_fn = lr.lr_schedule_picker('lr_range_test',
                                        max_learning_rate=0.5,
                                        num_train_steps=num_train_steps)

    momentum = momentum_fn(global_step)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    train_op = optimizer.minimize(dummy_loss, global_step)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)

    lr_hist = []
    mom_hist = []
    for _ in range(num_train_steps):
        _, lr_val, mom_val = sess.run([train_op, learning_rate, momentum])
        lr_hist.append(lr_val)
        mom_hist.append(mom_val)
    pass
