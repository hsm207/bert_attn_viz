import tensorflow as tf
import modeling

try:

    from run_classifier import model_fn_builder

except tf.flags.DuplicateFlagError:
    pass


def load_bert_model(output_dir,
                    bert_config_file='./model/uncased_L-12_H-768_A-12/bert_config.json',
                    init_checkpoint='./tuned_model/model.ckpt-2461',
                    num_labels=2,
                    attn_processor_fn=None):
    """
    Return's a pretrained BERT Estimator object
    :param output_dir: Directory to save the model's artifacts
    :param bert_config_file: Path to the model's bert_config.json file
    :param init_checkpoint: Checkpoint to use to initialize the model's weights
    :param num_labels: Number of labels that init_checkpoint was finetuned with
    :param attn_processor_fn:
    :return:
    """
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    tf.logging.info('Setting output dir to {} ...'.format(output_dir))
    # I don't expect to be running this model on a TPU so whatever...
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=8,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=init_checkpoint,
        learning_rate=1e-3,
        num_train_steps=2,
        num_warmup_steps=1,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=32,
        eval_batch_size=8,
        predict_batch_size=8,
        export_to_tpu=False,
        params={'attn_processor_fn': attn_processor_fn})

    return estimator
