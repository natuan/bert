from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

import json
import numpy as np
import pandas as pd
from session import get_session_info


##################### CONFIGURATION FOR JOB CLASSIFICATION #####################
config = get_session_info()

training_config = {}

training_config['TRAINING_SESSION_NAME'] = 'hidden20_drop05_05_lr05e-5'

training_config['DEBUGGING'] = False

training_config['BERT_BASE_DIR'] = os.path.join('base_models', 'uncased_L-12_H-768_A-12')

training_config['INIT_CHECKPOINT'] = os.path.join(training_config['BERT_BASE_DIR'], 'bert_model.ckpt')

#training_config['INIT_CHECKPOINT'] = '/home/tnguyen/src/bert/JUL06/outputs_train_jul10_drop05_lr03e-5/model.ckpt-9600.index'

training_config['EXTRA_HIDDEN_NEURONS'] = 20

training_config['EXTRA_HIDDEN_DROPOUT'] = 0.5

training_config['BASE_OUTPUT_DROPOUT'] = 0.5    # Default: 0.1

training_config['MAX_SEQ_LENGTH'] = 512    # Default: 128

training_config['TRAIN_BATCH_SIZE'] = 10   # max_seq_length 512: train_batch_size <= 10.
training_config['EVAL_BATCH_SIZE'] = 8     # Default: 8
training_config['PREDICT_BATCH_SIZE'] = 8  # Default: 8

training_config['LEARNING_RATE'] = 5e-5    # Default: 5e-5

training_config['NUM_TRAIN_EPOCHS'] = 2.0  # Default: 3.0

training_config['WARMUP_PROPOTION'] = 0.1  # Default: 0.1

# The evaluation will also be done every of the below checkpoint steps
# I chose 4000 because it's about the number of development data points
training_config['SAVE_CHECKPOINTS_STEPS'] = int(4000/training_config['TRAIN_BATCH_SIZE'])

training_config['ITERATIONS_PER_LOOP'] = 1000  # Default: 1000

training_config['SHUFFLE_BUFFER_SIZE'] = 10099  # Equal the training set size

training_config['LR_POLY_DECAY'] = {'name': 'polynomial_decay',
                                    'active': True,
                                    'decay_steps': None,  # TO BE UPDATED
                                    'end_learning_rate': 0.0,
                                    'power': 1.0,
                                    'cycle': False}

training_config['LR_PIECEWISE_CONST_DECAY'] = {'name': 'piecewise_constant_decay',
                                               'active': False,
                                               'boundaries': [2000, 3500],
                                               'values': [5e-5, 5e-6, 5e-7]}

#################################################################################

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "session", None, "The session name.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", training_config['INIT_CHECKPOINT'],
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", training_config['MAX_SEQ_LENGTH'],
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", training_config['TRAIN_BATCH_SIZE'], "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", training_config['EVAL_BATCH_SIZE'], "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", training_config['PREDICT_BATCH_SIZE'], "Total batch size for predict.")

flags.DEFINE_float("learning_rate", training_config['LEARNING_RATE'], "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", training_config['NUM_TRAIN_EPOCHS'],
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", training_config['WARMUP_PROPOTION'],
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", training_config['SAVE_CHECKPOINTS_STEPS'],
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", training_config['ITERATIONS_PER_LOOP'],
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class JobProcessor:
  """
  Processor for Scout's jobs
  """

  def get_train_examples(self):
    return self._get_examples('train')

  def get_dev_examples(self):
    return self._get_examples('dev')

  def get_test_examples(self):
    return self._get_examples('test')

  def get_labels(self):
    train_file_path = os.path.join(config['session_dir'], 'train.csv')
    df = pd.read_csv(train_file_path)
    level1_ids = np.unique(df['level1_id'].values)
    return level1_ids

  def _get_examples(self, set_type='train'):
    assert set_type in {'train', 'dev', 'test'}
    csv_file_path = os.path.join(config['session_dir'], f'{set_type}.csv')
    df = pd.read_csv(csv_file_path)

    if training_config['DEBUGGING']:
      df = df.iloc[:500]
    
    examples = []
    for job_id, job_text, level1_id in zip(df['job_id'], df['job_text'], df['level1_id']):
      guid = job_id
      text_a = tokenization.convert_to_unicode(job_text)
      text_b = None
      label = level1_id
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      # tnguyen: see https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#shuffle
      d = d.shuffle(buffer_size=training_config['SHUFFLE_BUFFER_SIZE'])

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def _neuron_layer(X, n_neurons, name, activation=None):
  n_inputs = int(X.get_shape()[1])
  with tf.variable_scope(name):
    W = tf.get_variable('weights', [n_inputs, n_neurons],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable('bias', [n_neurons], initializer=tf.zeros_initializer())
    z = tf.matmul(X, W) + b
    if activation == 'relu':
      return tf.nn.relu(z)
    else:
      return z

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  base_output_layer = model.get_pooled_output()
  if is_training:
    base_output_layer = tf.nn.dropout(base_output_layer,
                                      rate=training_config['BASE_OUTPUT_DROPOUT'])

  extra_hidden = _neuron_layer(base_output_layer, training_config['EXTRA_HIDDEN_NEURONS'],
                               name='extra_hidden_01',
                               activation='relu')
  if is_training:
    extra_hidden = tf.nn.dropout(extra_hidden,
                                 rate=training_config['EXTRA_HIDDEN_DROPOUT'])

  logits = _neuron_layer(extra_hidden, num_labels, name='output')
  with tf.variable_scope('cross-entropy'):
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     lr_decay_config, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, lr_decay_config, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def predict(estimator, examples, label_list, idx2label, tokenizer,
            output_tf_record_file_path,
            output_predicted_class_file_path):
  file_based_convert_examples_to_features(examples, label_list,
                                          FLAGS.max_seq_length, tokenizer,
                                          output_tf_record_file_path)
  predict_input_fn = file_based_input_fn_builder(
      input_file=output_tf_record_file_path,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)
  result = estimator.predict(input_fn=predict_input_fn)
  job_id = []
  label = []
  predicted_label = []

  for (idx, prediction) in enumerate(result):
    job_id.append(examples[idx].guid)
    label.append(examples[idx].label)
    probs = prediction['probabilities']
    predicted = idx2label[np.argmax(probs)]
    assert 1 <= predicted <= 26
    predicted_label.append(predicted)

  df = pd.DataFrame(data={'job_id': job_id,
                          'predicted_label': predicted_label,
                          'label': label})
  df.to_csv(output_predicted_class_file_path, index=False)


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config_file = os.path.join(training_config['BERT_BASE_DIR'], 'bert_config.json')
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  output_dir = os.path.join(config['session_dir'], f"outputs_{training_config['TRAINING_SESSION_NAME']}")
  if FLAGS.do_train:
    assert not os.path.exists(output_dir)
    tf.gfile.MakeDirs(output_dir)

  train_config_file_path = os.path.join(output_dir, 'training_config.json')
  with open(train_config_file_path, 'w') as f:
    json.dump(training_config, f)

  processor = JobProcessor()

  label_list = processor.get_labels()
  idx2label = {idx: label for idx, label in enumerate(label_list)}
  
  vocab_file = os.path.join(training_config['BERT_BASE_DIR'], 'vocab.txt')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = processor.get_train_examples()
  num_train_steps = int(
    len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  lr_decay_config = None
  if training_config['LR_POLY_DECAY']['active']:
    training_config['LR_POLY_DECAY']['decay_steps'] = num_train_steps
    lr_decay_config = training_config['LR_POLY_DECAY']
  elif training_config['LR_PIECEWISE_CONST_DECAY']['active']:
    assert training_config['LR_PIECEWISE_CONST_DECAY']['values'][0] == \
      training_config['LEARNING_RATE']
    lr_decay_config = training_config['LR_PIECEWISE_CONST_DECAY']
  else:
    assert False, 'Must specify a decay strategy'

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      lr_decay_config=lr_decay_config,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    # Set up training spec
    train_file = os.path.join(output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=num_train_steps)

    # Set up evaluation spec
    eval_examples = processor.get_dev_examples()
    eval_file = os.path.join(output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
      eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    eval_input_fn = file_based_input_fn_builder(
      input_file=eval_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

    eval_summary_hook = tf.train.SummarySaverHook(
      save_steps=100,  ## TODO: Why 100?
      output_dir=output_dir,
      scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))

    eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      hooks=[eval_summary_hook],
      steps=len(eval_examples) / FLAGS.eval_batch_size,
      start_delay_secs=10,
      throttle_secs=120)


    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  if FLAGS.do_eval:
    # Training set
    train_examples = processor.get_train_examples()
    train_tf_record_file_path = os.path.join(output_dir, 'train.tf_record')
    train_predicted_label_file_path = os.path.join(output_dir, 'train_predict.csv')
    predict(estimator, train_examples, label_list, idx2label, tokenizer,
            train_tf_record_file_path, train_predicted_label_file_path)

    # Dev set
    predict_examples = processor.get_dev_examples()
    output_tf_record_file_path = os.path.join(output_dir, 'dev.tf_record')
    output_predicted_label_file_path = os.path.join(output_dir, 'dev_predict.csv')
    predict(estimator, predict_examples, label_list, idx2label, tokenizer,
            output_tf_record_file_path, output_predicted_label_file_path)

    # Test set
    predict_examples = processor.get_test_examples()
    output_tf_record_file_path = os.path.join(output_dir, 'test.tf_record')
    output_predicted_label_file_path = os.path.join(output_dir, 'test_predict.csv')
    predict(estimator, predict_examples, label_list, idx2label, tokenizer,
            output_tf_record_file_path, output_predicted_label_file_path)

    
  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples()
    output_tf_record_file_path = os.path.join(output_dir, 'test.tf_record')
    output_predicted_label_file_path = os.path.join(output_dir, 'test_predict.csv')
    predict(estimator, predict_examples, label_list, idx2label, tokenizer,
            output_tf_record_file_path, output_predicted_label_file_path)

    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    output_predict_file = os.path.join(output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples

  print('******  TRAINING SESSION {} COMPLETES ******'.
        format(training_config['TRAINING_SESSION_NAME']))
  if training_config['DEBUGGING']:
    print('WARNING: DEBUGGING MODE IS ACTIVE')


if __name__ == "__main__":
  flags.mark_flag_as_required("session")
  tf.app.run()
