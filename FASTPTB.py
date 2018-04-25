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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import (
    ops,
    tensor_shape,
)
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import (
    array_ops,
    clip_ops,
    embedding_ops,
    init_ops,
    math_ops,
    nn_ops,
    partitioned_variables,
    variable_scope as vs,
    rnn_cell
)

from tensorflow.python.ops.math_ops import (
    sigmoid,
    tanh,
)
from tensorflow.python.layers import base as base_layer
from tensorflow.python.platform import tf_logging as logging



class LSTMCell(tf.nn.rnn_cell.RNNCell):


  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None, name=None):

   
    
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)
    self._scope = vs.get_variable_scope()
    
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    
    return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
 
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    h_depth = self._num_units
    self._kernel = self.add_variable(
        'weights',
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        'bias',
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state,scope=None):
    with vs.variable_scope(scope or type(self).__name__):  # "LSTMCell"
        one = constant_op.constant(1, dtype=dtypes.int32)
        with vs.variable_scope("Gates"):
          dtype = [a.dtype for a in [inputs, state]][0]
    # Parameters of gates are concatenated into one multiply for efficiency.
          if self._state_is_tuple:
            c, h = state
          else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

          gate_inputs = math_ops.matmul(
                array_ops.concat([inputs, h], 1), self._kernel)
          gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
          i, j, f, o = array_ops.split(
                value=gate_inputs, num_or_size_splits=4, axis=one)

          forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
          add = math_ops.add
          multiply = math_ops.multiply
          new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                       multiply(sigmoid(i), self._activation(j)))
          new_h = multiply(self._activation(new_c), sigmoid(o))

          if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
          else:
            new_state = array_ops.concat([new_c, new_h], 1)
	  print("ccccccc:",new_c)
          
    return new_h, new_state
class PTBInput(object):
    def __init__(self, config, data, name = None):
        
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name = name)



class PTBModel(object):
    def __init__(self, is_training, config, input_):
        print("PTBModel")
        self._input = input_
        batch_size = input_.batch_size
        
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
  	self.W_x = tf.Variable(tf.random_uniform(
             [size,size],
             -np.sqrt(2.0/size),
             np.sqrt(2.0/size)),
             dtype=tf.float32)
        self.b_x = tf.Variable(tf.zeros([size]),dtype=tf.float32)
	self.W_h = tf.Variable(
                initial_value=0.05 * np.identity(size),
                dtype=tf.float32)
	self.A = tf.zeros(
            [batch_size,size,size],
            dtype=tf.float32)
	self.gain = tf.Variable(tf.ones(
                [size]),
                dtype=tf.float32)
        self.bias = tf.Variable(tf.zeros(
                [size]),
                dtype=tf.float32)
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        attn_cell = lstm_cell

        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple = True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)


        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype = tf.float32)
	    
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
            
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:tf.get_variable_scope().reuse_variables() 
                
                (cell_output, state) = cell(inputs[:, time_step, :], state)
		
		self.h_s = tf.reshape(cell_output,[batch_size, 1, size])
		self.A = tf.add(tf.scalar_mul(0.95, self.A),
                    tf.scalar_mul(0.5, tf.matmul(tf.transpose(
                        self.h_s, [0, 2, 1]),self.h_s))) 
		
		for _ in range(5):
		    self.h_s= tf.reshape(
                        tf.matmul(inputs[:, time_step, :], self.W_x)+self.b_x,
                        tf.shape(self.h_s)) + tf.reshape(
                        tf.matmul(cell_output, self.W_h), tf.shape(self.h_s))+ \
                        tf.matmul(self.h_s, self.A)
		    mu = tf.reduce_mean(self.h_s, reduction_indices=0)
		    sigma = tf.sqrt(tf.reduce_mean(tf.square(self.h_s - mu),
                        reduction_indices=0))
		    self.h_s = tf.div(tf.multiply(self.gain, (self.h_s - mu)), sigma)
                    self.h_s = tf.nn.relu(self.h_s)
		cell_output = tf.reshape(self.h_s,[batch_size,size])
                outputs.append(cell_output)
	
	
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
	softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        
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
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
	    
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]

        state = vals["final_state"]

        costs += cost
        
        iters += model.input.num_steps
        
        if verbose and step % (model.input.epoch_size // 10) == 10:

            print ("%.3f perplexity: %.3f speed : %.0f wps" 
                %(step * 1.0 / model.input.epoch_size, np.exp(costs / iters), 
                iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = MediumConfig()
eval_config = MediumConfig()
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


