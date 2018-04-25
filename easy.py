from __future__ import division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers
import time 
import numpy as np
import tensorflow as tf
import reader
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _like_rnncell(cell):
  """Checks that a given object is an RNNCell by using duck typing."""
  conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
                hasattr(cell, "zero_state"), callable(cell)]
  return all(conditions)

def _zero_state_tensors(state_size, batch_size, dtype):
 
  def get_state_shape(s):
    a = _concat(batch_size, s)
    size = array_ops.zeros(a, dtype=dtype)
    if not context.executing_eagerly():
      a_static = _concat(batch_size, s, static=True)
      size.set_shape(a_static)
    return size
  return nest.map_structure(get_state_shape, state_size)
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("a", "h"))
def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.

  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).

  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.

  Returns:
    shape: the concatenation of prefix and suffix.

  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape


def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


@tf_export("nn.rnn_cell.LSTMStateTuple")
class RNNCell(base_layer.Layer):
  

  def __call__(self, inputs, state, scope=None):
    
    print("RNNCELL")
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      scope_attrname = "rnncell_scope"
      scope = getattr(self, scope_attrname, None)
      if scope is None:
        scope = vs.variable_scope(vs.get_variable_scope(),
                                  custom_getter=self._rnn_get_variable)
        setattr(self, scope_attrname, scope)
      with scope:
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.executing_eagerly():
      trainable = variable._trainable  # pylint: disable=protected-access
    else:
      trainable = (
          variable in tf_variables.trainable_variables() or
          (isinstance(variable, tf_variables.PartitionedVariable) and
           list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size, s]` for each s in `state_size`.
    """
    # Try to use the last cached zero_state. This is done to avoid recreating
    # zeros, especially when eager execution is enabled.
    #print("zero_state")
    state_size = self.state_size
    #print("state_size:",state_size)
    is_eager = context.executing_eagerly()
    if is_eager and hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and
          last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output


class LSTMStateTuple(_LSTMStateTuple):
  
  __slots__ = ()

  @property
  def dtype(self):
    (a, h) = self
    if a.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(a.dtype), str(h.dtype)))
    return a.dtype

@tf_export("nn.rnn_cell.BasicLSTMCell")

class LayerRNNCell(RNNCell):
  
  def __call__(self, inputs, state, scope=None, *args, **kwargs):
    
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    
    return base_layer.Layer.__call__(self, inputs, state, scope=scope,
                                     *args, **kwargs)
class FASTCell(LayerRNNCell):
  

  def __init__(self, num_units,num_steps,batch_size,vocab_size, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None, name=None):
        
        super(FASTCell, self).__init__(_reuse=reuse, name=name)
   
        if not state_is_tuple:
          logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
        self.input_spec = base_layer.InputSpec(ndim=2)
        print("num_units:",num_units)
        print("batch_size:",batch_size)
        self.outputsa = []
        self._batch_size = batch_size
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self.X = tf.placeholder(tf.float32,
            shape=[None,num_steps,vocab_size], name='inputs_X')
        self.y = tf.placeholder(tf.float32,
            shape=[None,vocab_size], name='targets_y')
        self.l = tf.placeholder(tf.float32, [], # need [] for tf.scalar_mul
            name="learning_rate")
        self.e = tf.placeholder(tf.float32, [],
            name="decay_rate")
        with tf.variable_scope("fast_weights"):

            # input weights (proper initialization)
            self.W_x = tf.Variable(tf.random_uniform(
                [vocab_size, num_units],
                -np.sqrt(2.0/vocab_size),
                np.sqrt(2.0/vocab_size)),
                dtype=tf.float32)
            self.b_x = tf.Variable(tf.zeros(
                [num_units]),
                dtype=tf.float32)

            # hidden weights (See Hinton's video @ 21:20)
            self.W_h = tf.Variable(
                initial_value=0.05 * np.identity(num_units),
                dtype=tf.float32)
            # softmax weights (proper initialization)
            self.W_softmax = tf.Variable(tf.random_uniform(
                [num_units,vocab_size],
                -np.sqrt(2.0 /num_units),
                np.sqrt(2.0 / num_units)),
                dtype=tf.float32)
            self.b_softmax = tf.Variable(tf.zeros(
                [vocab_size]),
                dtype=tf.float32)

            # scale and shift for layernorm
            self.gain = tf.Variable(tf.ones(
                [num_units]),
                dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(
                [num_units]),
                dtype=tf.float32)
        # fast weights and hidden state initialization
        self.A = tf.zeros(
            [batch_size, num_units, num_units],
            dtype=tf.float32)
        self.h = tf.zeros(
            [batch_size, num_units],
            dtype=tf.float32)
  @property
  def state_size(self):
    
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units
        
    
  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * self.state_size]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    
    for time_step in range(20):
        print("time_step=",time_step)
        if time_step > 0:tf.get_variable_scope().reuse_variables()
               # Parameters of gates are concatenated into one multiply for efficiency.
        self.h = tf.nn.relu((tf.matmul(self.X[:, time_step, :], self.W_x)+self.b_x) +
                (tf.matmul(self.h, self.W_h)))
        self.h_s = tf.reshape(self.h,
                    [self._batch_size, 1, self._num_units])
        
        self.A = tf.add(tf.scalar_mul(self.l, self.A),
                    tf.scalar_mul(self.e, tf.matmul(tf.transpose(
                        self.h_s, [0, 2, 1]), self.h_s)))
        
    
        for _ in range(1):
            self.h_s = tf.reshape(
                tf.matmul(self.X[:,time_step, :], self.W_x)+self.b_x,
                tf.shape(self.h_s)) + tf.reshape(
                tf.matmul(self.h, self.W_h), tf.shape(self.h_s)) + \
                tf.matmul(self.h_s, self.A)
            
                # Apply layernorm
            mu = tf.reduce_mean(self.h_s, reduction_indices=0) # each sample
            sigma = tf.sqrt(tf.reduce_mean(tf.square(self.h_s - mu),
                reduction_indices=0))
            #print("self.h_s =",self.h_s )
            #print("self.gain =",self.gain)
            #print("q=",tf.multiply(self.gain, (self.h_s - mu)))
            #print("sigma=",sigma)
            self.h_s = tf.div(tf.multiply(self.gain, (self.h_s - mu)), sigma) + \
            self.bias

            # Apply nonlinearity
            self.h_s = tf.nn.relu(self.h_s)

            # Reshape h_s into h
            self.h = tf.reshape(self.h_s,[self._batch_size, self._num_units])
            self.outputsa.append(self.h)
    new_a = self.A
    new_h = self.outputsa
    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_a, new_h)
    else:
      new_state = array_ops.concat([new_a, new_h], 1)

    return new_h, new_state  
   


class PTBInput(object):
    def __init__(self, config, data, name = None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name = name)


class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        
	def lstm_cell():
            return FASTCell(size,num_steps,batch_size,vocab_size,forget_bias = 0.0, state_is_tuple = True)
        attn_cell = lstm_cell
	cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple = True)
	self._initial_state = cell.zero_state(batch_size, tf.float32)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)


        print("inputs=",inputs)
        state = self._initial_state
        with tf.variable_scope("RNN"):
            tf.get_variable_scope().reuse_variables() 
            (cell_output, state) = cell(inputs[:,1, :], state)
            
	print("1234567")
        output = tf.reshape(tf.concat(cell_output, 1), [-1, size])
        print("size:",size)
	print("output:",output)
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        print("logits=",logits)
        print("input_.targets=",input_.targets)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])],[tf.ones([batch_size * num_steps], dtype = tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return


        self._lr = tf.Variable(0.0, trainable = False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape = [], name = "new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict = {self._new_lr: lr_value})



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

class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
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


class LargeConfig(object):
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



class TestConfig(object):
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

def run_epoch(session, model, eval_op = None, verbose = False):
    print("run_epoch")
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        print("infor")
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[a] = state[i].a
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]

        state = vals["final_state"]

        costs += cost
        # print cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:

            print ("%.3f perplexity: %.3f speed : %.0f wps" 
                %(step * 1.0 / model.input.epoch_size, np.exp(costs / iters), 
                iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1


with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config = config, data = train_data, name = 'TrainInput')
        with tf.variable_scope("Model", reuse = None, initializer = initializer):
            m = PTBModel(is_training = True, config = config, input_ = train_input)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config = config, data = valid_data, name = "ValidInput")
        with tf.variable_scope("Model", reuse = True, initializer = initializer):
            mvalid = PTBModel(is_training = False, config = config, input_ = valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config = eval_config, data = test_data, name = "TestInput")

        with tf.variable_scope("Model", reuse = True, initializer = initializer):
            mtest = PTBModel(is_training = False, config = eval_config, input_ = test_input)


        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" %(i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op = m.train_op, verbose = True)
                print("Epoch: %d Train Perplexity: %.3f" %(i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d valid Perplexity: %.3f" %(i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" %test_perplexity)
