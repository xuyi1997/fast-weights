# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights权重初始值
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient最大允许梯度值
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM#LSTM展开步骤的数量
- hidden_size - the number of LSTM units隐藏单元数量
- max_epoch - the number of epochs trained with the initial learning rate以初始学习率训练的epoch数目
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

#定义语言模型处理的输入数据class
class PTBInput(object):
  """The input data."""
#初始化，读取config中的batch_size,num_steps到本地变量
  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps#num_steps是LSTM的展开步数
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps#计算每个epoch内需要多少轮训练的迭代
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""
#训练标记，配置参数
  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size#hiddensize是LSTM的隐藏节点数
    vocab_size = config.vocab_size#词汇表

    with tf.device("/cpu:0"):
## 设定embedding变量以及转化输入单词为embedding里的词向量（embedding_lookup函数）
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
#用tf.nn.embedding_lookup查询单词对应的向量表达获得inputs
    if is_training and config.keep_prob < 1:
# 如果为训练状态，则添加一层对输入进行dropout
      inputs = tf.nn.dropout(inputs, config.keep_prob)
# 从state开始运行RNN架构，输出为output的输出以及新的state
    output, state = self._build_rnn_graph(inputs, config, is_training)
 # 输出定义为output的输出乘以权重softmax weight w后加上偏置softmax bias b. 这被叫做logit
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
    # loss函数是average negative log probability, 这里我们有现成的函数sequence_loss
    # 来达到这个效果。
    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost汇总batch的偏差
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return#如果此时不是训练状态直接返回

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()#使用tf.trainable_variables获取所有可训练的参数tvars
# 根据张量间的和的norm来clip多个张量，控制梯度的最大范数，某种程度上起到正则化的效果。Gradient Clipping可防止Gradient Explosion梯度爆炸的问题，如果对梯度不加以限制，则可能会因为迭代中梯度过大导致难以收敛
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
# 用之前的变量learning rate来起始梯度下降优化器。 #再创建训练操作_train_op，用optimizer.apply_gradients将前面clip过的梯度应用到所有可训练的参数tvars上，
    #然后使用tf.train.get_or_create_global_step()生成全局统一的训练步数
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())
#设置一个_new_lr的placeholder用以控制学习速率。
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
#同时定义个assign_lr的函数，用以在外部控制模型的学习速率
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
#使用tf.contrib.rnn.BasicLSTMCell设置默认的lstm单元
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
#使用rnn堆叠函数tf.contrib.rnn.MultiRNNCell将前面的lstmcell多层堆叠得到cell，堆叠次数为config的numlayers
    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state
#定义个assign_lr的函数，用以在外部控制模型的学习速率方式是将学习速率值传入_new_lr这个place_holder,并执行_update_lr操作完成对学习速率的修改
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)
#python中的@property装饰器可以将返回变量设为只读，防止修改变量引发的问题

  #这里定义input,initial_state,cost,lr,final_state,train_op为property，方便外部访问
  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name
#接下来定义几种不同大小的模型的参数
#首先是小模型的设置

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1#网络中权重值的初始scale
  learning_rate = 1.0#学习速率的初始值
  max_grad_norm = 5#前面提到的梯度的最大范数
  num_layers = 2# num_layers是LSTM可以堆叠的层数
  num_steps = 20#是LSTM梯度反向传播的展开步数
  hidden_size = 200#LSTM的隐含节点数
  max_epoch = 4#是初始学习速率的可训练的epoch数，在此之后需要调整学习速率
  max_max_epoch = 13 #总共可以训练的epoch数
  keep_prob = 1.0#keep_prob是dorpout层的保留节点的比例
  lr_decay = 0.5 #学习速率的衰减速率
  batch_size = 20#每个batch中样本的数量
  vocab_size = 10000
  rnn_mode = BLOCK

#具体每个参数的值，在不同的配置中对比才有意义

#在中等模型中，我们减小了init_state，即希望权重初值不要过大，小一些有利于温和的训练
#学习速率和最大梯度范数不变，LSTM层数不变。
#这里将梯度反向传播的展开步数从20增大到35.
#hidden_size和max_max_epoch也相应地增大约3倍;同时，这里开始设置dropout的keep_prob到0.5
#而之前设置1,即没有dropout;
#因为学习迭代次数的增大，因此将学习速率的衰减速率lr_decay也减小了。
#batch_size和词汇表vocab_size的大小保持不变
class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
#大型模型，进一步缩小了init_scale并大大放宽了最大梯度范数max_grad_norm到10
#同时将hidden_size提升到了1500,并且max_epoch,max_max_epoch也相应增大了;
#而keep_drop也因为模型复杂度的上升继续下降，学习速率的衰减速率lr_decay也进一步减小

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
#定义训练一个epoch数据的函数run_epoch。

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""# 记录当前时间，初始化损失costs和迭代数iters。
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
##并执行model.initial_state来初始化状态并获得初始状态

  #接着创建输出结果的字典表fetches
  #其中包括cost和final_state
  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op
 # 接着进行循环训练中，次数为epoch_size
  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
# 将全部的LSTM单元的state加入feed_dict，然后传入feed_dict并执行
    # fetchees对网络进行一次训练，并且拿到cost和state
    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
#我们累加cost到costs，并且累加num_steps到iters。
    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))
# 最后返回perplexity作为函数的结果
  return np.exp(costs / iters)


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config


def main(_):
 # 需要首先确认输入数据的path，不然没法训练模型
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))
#使用reader.ptb_raw_data直接读取解压后的数据，得到训练数据，验证数据和测试数据
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data
# 读取用户输入的config，这里决定了用是小，中还是大模型
  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
#先进行initialization
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
#使用PTBInput和PTBModel创建一个用来训练的模型
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)
# 使用PTBInput和PTBModel创建一个用来验证的模型mvalid
    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)
# 使用PTBInput和PTBModel创建一个用来验证的模型Test
    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)
#其中训练和验证模型直接使用前面的cofig，测试模型则使用前面的测试配置eval_config
    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
#使用tf.train.Supervisor创建训练的管理器sv

  #并使用sv.managed_session()创建默认的session
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      #再执行训练多个epoch数据的循环
      for i in range(config.max_max_epoch):
 #在每个epoch循环内，我们先计算累计的学习速率衰减值，
      #这里只需要计算超过max_epoch的轮数，再求lr_decay的超出轮数次幂即可
      #然后将初始学习速率乘以累计的衰减，并更新学习速率。
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
#在循环内执行一个epoch的训练和验证，并输出当前的学习速率，训练和验证即的perplexity
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
#在完成全部训练之后，计算并输出模型在测试集上的perplexity
      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
	tf.app.run()
