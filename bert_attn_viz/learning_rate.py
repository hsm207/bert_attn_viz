import tensorflow as tf


def lr_schedule_picker(lr_schedule_type='default', **kwargs):
    supported_lr_schedules = ['default', 'lr_range_test', 'one_cycle']
    assert lr_schedule_type in supported_lr_schedules, \
        "{} is not a supported learning rate schedule".format(lr_schedule_type)

    if lr_schedule_type == 'default':
        lr_schedule = default_learning_rate_schedule(init_lr=kwargs['init_lr'],
                                                     num_train_steps=kwargs['num_train_steps'],
                                                     num_warmup_steps=kwargs['num_warmup_steps'])
    elif lr_schedule_type == 'lr_range_test':
        lr_schedule = lr_range_test(max_learning_rate=kwargs['max_learning_rate'],
                                    num_train_steps=kwargs['num_train_steps'])

    return lr_schedule


def default_learning_rate_schedule(init_lr, num_train_steps, num_warmup_steps):
    """
    Returns a callable that will create the default learning rate schedule which is to linearly increase the learning
    rate from 0 to init_lr in num_warmup steps, and then linearly decrease it to 0 for the remaining steps
    :param init_lr: a float representing the max learning rate to use
    :param num_train_steps: total number of iterations to use in this training run
    :param num_warmup_steps: total number of steps (out of num_train_steps) to linearly increase the learning rate
    :return: A callable that takes the global step and returns a tensor representing the appropriately scheduled
            learning rate
    """

    def lr_scheduler(global_step):
        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if num_warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                    (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        return learning_rate

    return lr_scheduler


def lr_range_test(max_learning_rate, num_train_steps):
    """
    Returns a callable that will return a learning rate schedule that will do a LR range test which is linearly
    increase the learning rate from 0 to max_learning_rate throughout a training run.
    See https://arxiv.org/pdf/1803.09820.pdf for details.

    :param max_learning_rate: A float representing the max learning rate to try
    :param num_train_steps: Number of iterations to run the LR range test
    :return: A callable that will take the global step and return a tensor representing the appropriately scheduled
             learning rate
    """

    def lr_scheduler(global_step):
        start_lr = tf.constant(0, tf.float32)
        end_lr = tf.constant(max_learning_rate, tf.float32)
        lr_increment = (end_lr - start_lr) / (num_train_steps - 1)

        global_step_float = tf.cast(global_step, tf.float32)

        learning_rate = start_lr + global_step_float * lr_increment

        return learning_rate

    return lr_scheduler
