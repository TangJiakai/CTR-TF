import tensorflow as tf
from tensorflow.python import keras
from main_code.layer.din import ActivationUnit
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras import activations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import base_layer
import numpy as np

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

_concat = rnn_cell_impl._concat


class DIENAttentionLayer(keras.layers.Layer):
    def __init__(self, attn_hidden_units, l2_reg, **kwargs):
        self.attn_hidden_units = attn_hidden_units
        self.l2_reg = l2_reg
        super(DIENAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attn_net = ActivationUnit(self.attn_hidden_units, self.l2_reg)
        self.grucell = AUGRUCell(num_units=input_shape[0][-1])
        # self.gru = GRU(units=input_shape[0][-1], kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        # self.gru = keras.layers.GRU(units=input_shape[0][-1], kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        super().build(input_shape)

    # @tf.autograph.experimental.do_not_convert
    def call(self, inputs, mask=None, hist_len=None, training=None, **kwargs):
        attn_scores = self.attn_net(inputs, training=training)
        query, key = inputs
        padding = tf.ones_like(attn_scores) * (-2 ** 32 + 1)
        attn_scores = tf.where(mask, attn_scores, padding)
        output, hidden_state = dynamic_rnn(self.grucell, inputs=key, attn_scores=attn_scores, sequence_length=tf.squeeze(hist_len), dtype=tf.float32, scope=self.name)
        return hidden_state

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {'attn_hidden_units': self.attn_hidden_units, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class _Linear(object):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.



    Args:

      args: a 2D Tensor or a list of 2D, batch x n, Tensors.

      output_size: int, second dimension of weight variable.

      dtype: data type for variables.

      build_bias: boolean, whether to build a bias variable.

      bias_initializer: starting value to initialize the bias

        (default is all zeros).

      kernel_initializer: starting value to initialize the weight.



    Raises:

      ValueError: if inputs_shape is wrong.

    """

    def __init__(self,

                 args,

                 output_size,

                 build_bias,

                 bias_initializer=None,

                 kernel_initializer=None):

        self._build_bias = build_bias

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")

        if not nest.is_sequence(args):

            args = [args]

            self._is_sequence = False

        else:

            self._is_sequence = True

        # Calculate the total size of arguments on dimension 1.

        total_arg_size = 0

        shapes = [a.get_shape() for a in args]

        for shape in shapes:

            if shape.ndims != 2:
                raise ValueError(
                    "linear is expecting 2D arguments: %s" % shapes)

            if shape[1] is None:

                raise ValueError("linear expects shape[1] to be provided for shape %s, "

                                 "but saw %s" % (shape, shape[1]))

            else:

                total_arg_size += int(shape[1])#.value

        dtype = [a.dtype for a in args][0]

        scope = vs.get_variable_scope()

        with vs.variable_scope(scope) as outer_scope:

            self._weights = vs.get_variable(

                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],

                dtype=dtype,

                initializer=kernel_initializer)

            if build_bias:

                with vs.variable_scope(outer_scope) as inner_scope:

                    inner_scope.set_partitioner(None)

                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(
                            0.0, dtype=dtype)

                    self._biases = vs.get_variable(

                        _BIAS_VARIABLE_NAME, [output_size],

                        dtype=dtype,

                        initializer=bias_initializer)

    def __call__(self, args):

        if not self._is_sequence:
            args = [args]

        if len(args) == 1:

            res = math_ops.matmul(args[0], self._weights)

        else:

            res = math_ops.matmul(array_ops.concat(args, 1), self._weights)

        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)

        return res


class AUGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:

      num_units: int, The number of units in the GRU cell.

      activation: Nonlinearity to use.  Default: `tanh`.

      reuse: (optional) Python boolean describing whether to reuse variables

       in an existing scope.  If not `True`, and the existing scope already has

       the given variables, an error is raised.

      kernel_initializer: (optional) The initializer to use for the weight and

      projection matrices.

      bias_initializer: (optional) The initializer to use for the bias.

    """

    def __init__(self,

                 num_units,

                 activation=None,

                 reuse=None,

                 kernel_initializer=None,

                 bias_initializer=None):

        super(AUGRUCell, self).__init__(_reuse=reuse)

        self._num_units = num_units

        self._activation = activation or math_ops.tanh

        self._kernel_initializer = kernel_initializer

        self._bias_initializer = bias_initializer

        self._gate_linear = None

        self._candidate_linear = None

    @property
    def state_size(self):

        return self._num_units

    @property
    def output_size(self):

        return self._num_units

    def __call__(self, inputs, state, att_score):

        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""

        if self._gate_linear is None:

            bias_ones = self._bias_initializer

            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(
                    1.0, dtype=inputs.dtype)

            with vs.variable_scope("gates"):  # Reset gate and update gate.

                self._gate_linear = _Linear(

                    [inputs, state],

                    2 * self._num_units,

                    True,

                    bias_initializer=bias_ones,

                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(

                    [inputs, r_state],

                    self._num_units,

                    True,

                    bias_initializer=self._bias_initializer,

                    kernel_initializer=self._kernel_initializer)

        c = self._activation(self._candidate_linear([inputs, r_state]))

        u = (1.0 - att_score) * u

        new_h = u * state + (1 - u) * c

        return new_h, new_h


def _check_supported_dtypes(dtype):
  if dtype is None:
    return
  dtype = dtypes.as_dtype(dtype)
  if not (dtype.is_floating or dtype.is_complex):
    raise ValueError("RNN cell only supports floating point inputs, "
                     "but saw dtype: %s" % dtype)


def _check_rnn_cell_input_dtypes(inputs):
  """Check whether the input tensors are with supported dtypes.
  Default RNN cells only support floats and complex as its dtypes since the
  activation function (tanh and sigmoid) only allow those types. This function
  will throw a proper error message if the inputs is not in a supported type.
  Args:
    inputs: tensor or nested structure of tensors that are feed to RNN cell as
      input or state.
  Raises:
    ValueError: if any of the input tensor are not having dtypes of float or
      complex.
  """
  for t in nest.flatten(inputs):
    _check_supported_dtypes(t.dtype)


def dynamic_rnn(cell,
                inputs,
                sequence_length=None,
                attn_scores=None, 
                initial_state=None,
                dtype=None,
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.
  Performs fully dynamic unrolling of `inputs`.
  Example:
  ```python
  # create a BasicRNNCell
  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.compat.v1.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```
  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)
  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.nn.rnn_cell.LSTMStateTuple for each cell
  outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```
  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      If `time_major == True`, this must be a `Tensor` of shape: `[max_time,
        batch_size, ...]`, or a nested tuple of such elements. This may also be
        a (possibly nested) tuple of Tensors satisfying this property.  The
        first two dimensions must match across all the inputs, but otherwise the
        ranks and other shape components may differ. In this case, input to
        `cell` at each time-step will replicate the structure of these tuples,
        except for the time dimension (from which the time is taken). The input
        to `cell` at each time step will be a `Tensor` or (possibly nested)
        tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`. Used
      to copy-through state and zero-out outputs when past a batch element's
      sequence length.  This parameter enables users to extract the last valid
      state and properly padded outputs, so it is provided for correctness.
    initial_state: (optional) An initial state for the RNN. If `cell.state_size`
      is an integer, this must be a `Tensor` of appropriate type and shape
      `[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
      should be a tuple of tensors having shapes `[batch_size, s] for s in
      cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency and
      can be run in parallel, will be.  This parameter trades off time for
      space.  Values >> 1 use more memory but take less time, while smaller
      values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs which
      would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
      these `Tensors` must be shaped `[max_time, batch_size, depth]`. If false,
      these `Tensors` must be shaped `[batch_size, max_time, depth]`. Using
      `time_major = True` is a bit more efficient because it avoids transposes
      at the beginning and end of the RNN calculation.  However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".
  Returns:
    A pair (outputs, state) where:
    outputs: The RNN output `Tensor`.
      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.
      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.
      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.
    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.
  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
  rnn_cell_impl.assert_like_rnncell("cell", cell)

  with vs.variable_scope(scope or "rnn") as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
      # (B,T,D) => (T,B,D)
      flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
      flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
      sequence_length = math_ops.cast(sequence_length, dtypes.int32)
      if sequence_length.get_shape().rank not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size, "
            "but saw shape: %s" % sequence_length.get_shape())
      sequence_length = array_ops.identity(  # Just to find it in the graph.
          sequence_length,
          name="sequence_length")

    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      if getattr(cell, "get_initial_state", None) is not None:
        state = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
      else:
        state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)), [
              "Expected shape for Tensor %s is " % x.name, packed_shape,
              " but saw shape: ", x_shape
          ])

    if not context.executing_eagerly() and sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _dynamic_rnn_loop(
        cell,
        inputs,
        state,
        attn_scores=attn_scores,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      outputs = nest.map_structure(_transpose_batch_time, outputs)

    return (outputs, final_state)


def _should_cache():
  """Returns True if a default caching device should be set, otherwise False."""
  if context.executing_eagerly():
    return False
  # Don't set a caching device when running in a loop, since it is possible that
  # train steps could be wrapped in a tf.while_loop. In that scenario caching
  # prevents forward computations in loop iterations from re-reading the
  # updated weights.
  ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  return control_flow_util.GetContainingWhileContext(ctxt) is None


def _transpose_batch_time(x):
  """Transposes the batch and time dimensions of a Tensor.
  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.
  Args:
    x: A Tensor.
  Returns:
    x transposed along the first two dimensions.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.rank is not None and x_static_shape.rank < 2:
    return x

  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape(
          [x_static_shape.dims[1].value,
           x_static_shape.dims[0].value]).concatenate(x_static_shape[2:]))
  return x_t


def _best_effort_input_batch_size(flat_input):
  """Get static input batch size if available, with fallback to the dynamic one.
  Args:
    flat_input: An iterable of time major input Tensors of shape `[max_time,
      batch_size, ...]`. All inputs should have compatible batch sizes.
  Returns:
    The batch size in Python integer if available, or a scalar Tensor otherwise.
  Raises:
    ValueError: if there is any input with an invalid shape.
  """
  for input_ in flat_input:
    shape = input_.shape
    if shape.rank is None:
      continue
    if shape.rank < 2:
      raise ValueError("Expected input tensor %s to have rank at least 2" %
                       input_)
    batch_size = shape.dims[1].value
    if batch_size is not None:
      return batch_size
  # Fallback to the dynamic batch size of the first input.
  return array_ops.shape(flat_input[0])[1]



def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      attn_scores,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
  """Internal implementation of Dynamic RNN.
  Args:
    cell: An instance of RNNCell.
    inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
      tuple of such elements.
    initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
      `cell.state_size` is a tuple, then this should be a tuple of tensors
      having shapes `[batch_size, s] for s in cell.state_size`.
    parallel_iterations: Positive Python int.
    swap_memory: A Python boolean
    sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
    dtype: (optional) Expected dtype of output. If not specified, inferred from
      initial_state.
  Returns:
    Tuple `(final_outputs, final_state)`.
    final_outputs:
      A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
      `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
      objects, then this returns a (possibly nested) tuple of Tensors matching
      the corresponding shapes.
    final_state:
      A `Tensor`, or possibly nested tuple of Tensors, matching in length
      and shapes to `initial_state`.
  Raises:
    ValueError: If the input depth cannot be inferred via shape inference
      from the inputs.
    ValueError: If time_step is not the same for all the elements in the
      inputs.
    ValueError: If batch_size is not the same for all the elements in the
      inputs.
  """
  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = nest.flatten(inputs)
  flat_output_size = nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = array_ops.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = _best_effort_input_batch_size(flat_input)

  inputs_got_shape = tuple(
      input_.get_shape().with_rank_at_least(3) for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape.dims[0].value
    got_batch_size = shape.dims[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _concat(batch_size, size)
    return array_ops.zeros(
        array_ops.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(
      _create_zero_arrays(output) for output in flat_output_size)
  zero_output = nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = math_ops.reduce_min(sequence_length)
    max_sequence_length = math_ops.reduce_max(sequence_length)
  else:
    max_sequence_length = time_steps

  time = array_ops.constant(0, dtype=dtypes.int32, name="time")

  with ops.name_scope("dynamic_rnn") as scope:
    base_name = scope

  def _create_ta(name, element_shape, dtype):
    return tensor_array_ops.TensorArray(
        dtype=dtype,
        size=time_steps,
        element_shape=element_shape,
        tensor_array_name=base_name + name)

  in_graph_mode = not context.executing_eagerly()
  if in_graph_mode:
    output_ta = tuple(
        _create_ta(
            "output_%d" % i,
            element_shape=(
                tensor_shape.TensorShape([const_batch_size]).concatenate(
                    _maybe_tensor_shape_from_tensor(out_size))),
            dtype=_infer_state_dtype(dtype, state))
        for i, out_size in enumerate(flat_output_size))
    input_ta = tuple(
        _create_ta(
            "input_%d" % i,
            element_shape=flat_input_i.shape[1:],
            dtype=flat_input_i.dtype)
        for i, flat_input_i in enumerate(flat_input))
    input_ta = tuple(
        ta.unstack(input_) for ta, input_ in zip(input_ta, flat_input))
  else:
    output_ta = tuple([0 for _ in range(time_steps.numpy())]
                      for i in range(len(flat_output_size)))
    input_ta = flat_input

  def _time_step(time, output_ta_t, state, attn_scores):
    """Take a time step of the dynamic RNN.
    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.
    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """

    if in_graph_mode:
      input_t = tuple(ta.read(time) for ta in input_ta)
      # Restore some shape information
      for input_, shape in zip(input_t, inputs_got_shape):
        input_.set_shape(shape[1:])
    else:
      input_t = tuple(ta[time.numpy()] for ta in input_ta)

    input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    # Keras RNN cells only accept state as list, even if it's a single tensor.
    is_keras_rnn_cell = _is_keras_rnn_cell(cell)
    if is_keras_rnn_cell and not nest.is_sequence(state):
      state = [state]

    attn_score = attn_scores[:, time, :]
    call_cell = lambda: cell(input_t, state, attn_score)

    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()

    # Keras cells always wrap state as list, even if it's a single tensor.
    if is_keras_rnn_cell and len(new_state) == 1:
      new_state = new_state[0]
    # Pack state if using state tuples
    output = nest.flatten(output)

    if in_graph_mode:
      output_ta_t = tuple(
          ta.write(time, out) for ta, out in zip(output_ta_t, output))
    else:
      for ta, out in zip(output_ta_t, output):
        ta[time.numpy()] = out

    return (time + 1, output_ta_t, new_state, attn_scores)

  if in_graph_mode:
    # Make sure that we run at least 1 step, if necessary, to ensure
    # the TensorArrays pick up the dynamic shape.
    loop_bound = math_ops.minimum(time_steps,
                                  math_ops.maximum(1, max_sequence_length))
  else:
    # Using max_sequence_length isn't currently supported in the Eager branch.
    loop_bound = time_steps

  _, output_final_ta, final_state, _ = control_flow_ops.while_loop(
      cond=lambda time, *_: time < loop_bound,
      body=_time_step,
      loop_vars=(time, output_ta, state, attn_scores),
      parallel_iterations=parallel_iterations,
      maximum_iterations=time_steps,
      swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  if in_graph_mode:
    final_outputs = tuple(ta.stack() for ta in output_final_ta)
    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
      shape = _concat([const_time_steps, const_batch_size],
                      output_size,
                      static=True)
      output.set_shape(shape)
  else:
    final_outputs = output_final_ta

  final_outputs = nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=final_outputs)
  if not in_graph_mode:
    final_outputs = nest.map_structure_up_to(
        cell.output_size, lambda x: array_ops.stack(x, axis=0), final_outputs)

  return (final_outputs, final_state)


def _infer_state_dtype(explicit_dtype, state):
  """Infer the dtype of an RNN state.
  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.
  Returns:
    dtype: inferred dtype of hidden state.
  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
  if explicit_dtype is not None:
    return explicit_dtype
  elif nest.is_sequence(state):
    inferred_dtypes = [element.dtype for element in nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all(x == inferred_dtypes[0] for x in inferred_dtypes)
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype


def _maybe_tensor_shape_from_tensor(shape):
  if isinstance(shape, ops.Tensor):
    return tensor_shape.as_shape(tensor_util.constant_value(shape))
  else:
    return shape


def _is_keras_rnn_cell(rnn_cell):
  """Check whether the cell is a Keras RNN cell.
  The Keras RNN cell accept the state as a list even the state is a single
  tensor, whereas the TF RNN cell does not wrap single state tensor in list.
  This behavior difference should be unified in future version.
  Args:
    rnn_cell: An RNN cell instance that either follow the Keras interface or TF
      RNN interface.
  Returns:
    Boolean, whether the cell is an Keras RNN cell.
  """
  # Cell type check is not strict enough since there are cells created by other
  # library like Deepmind that didn't inherit tf.nn.rnn_cell.RNNCell.
  # Keras cells never had zero_state method, which was from the original
  # interface from TF RNN cell.
  return (not isinstance(rnn_cell, rnn_cell_impl.RNNCell) and
          isinstance(rnn_cell, base_layer.Layer) and
          getattr(rnn_cell, "zero_state", None) is None)


def _rnn_step(time,
              sequence_length,
              min_sequence_length,
              max_sequence_length,
              zero_output,
              state,
              call_cell,
              state_size,
              skip_conditionals=False):
  """Calculate one step of a dynamic RNN minibatch.
  Returns an (output, state) pair conditioned on `sequence_length`.
  When skip_conditionals=False, the pseudocode is something like:
  if t >= max_sequence_length:
    return (zero_output, state)
  if t < min_sequence_length:
    return call_cell()
  # Selectively output zeros or output, old state or new state depending
  # on whether we've finished calculating each row.
  new_output, new_state = call_cell()
  final_output = np.vstack([
    zero_output if time >= sequence_length[r] else new_output_r
    for r, new_output_r in enumerate(new_output)
  ])
  final_state = np.vstack([
    state[r] if time >= sequence_length[r] else new_state_r
    for r, new_state_r in enumerate(new_state)
  ])
  return (final_output, final_state)
  Args:
    time: int32 `Tensor` scalar.
    sequence_length: int32 `Tensor` vector of size [batch_size].
    min_sequence_length: int32 `Tensor` scalar, min of sequence_length.
    max_sequence_length: int32 `Tensor` scalar, max of sequence_length.
    zero_output: `Tensor` vector of shape [output_size].
    state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
      or a list/tuple of such tensors.
    call_cell: lambda returning tuple of (new_output, new_state) where
      new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
      new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
    state_size: The `cell.state_size` associated with the state.
    skip_conditionals: Python bool, whether to skip using the conditional
      calculations.  This is useful for `dynamic_rnn`, where the input tensor
      matches `max_sequence_length`, and using conditionals just slows
      everything down.
  Returns:
    A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
      final_output is a `Tensor` matrix of shape [batch_size, output_size]
      final_state is either a single `Tensor` matrix, or a tuple of such
        matrices (matching length and shapes of input `state`).
  Raises:
    ValueError: If the cell returns a state tuple whose length does not match
      that returned by `state_size`.
  """

  # Convert state to a list for ease of use
  flat_state = nest.flatten(state)
  flat_zero_output = nest.flatten(zero_output)

  # Vector describing which batch entries are finished.
  copy_cond = time >= sequence_length

  def _copy_one_through(output, new_output):
    # TensorArray and scalar get passed through.
    if isinstance(output, tensor_array_ops.TensorArray):
      return new_output
    if output.shape.rank == 0:
      return new_output
    # Otherwise propagate the old or the new value.
    with ops.colocate_with(new_output):
      return array_ops.where(copy_cond, output, new_output)

  def _copy_some_through(flat_new_output, flat_new_state):
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    flat_new_output = [
        _copy_one_through(zero_output, new_output)
        for zero_output, new_output in zip(flat_zero_output, flat_new_output)
    ]
    flat_new_state = [
        _copy_one_through(state, new_state)
        for state, new_state in zip(flat_state, flat_new_state)
    ]
    return flat_new_output + flat_new_state

  def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()

    nest.assert_same_structure(zero_output, new_output)
    nest.assert_same_structure(state, new_state)

    flat_new_state = nest.flatten(new_state)
    flat_new_output = nest.flatten(new_output)
    return control_flow_ops.cond(
        # if t < min_seq_len: calculate and return everything
        time < min_sequence_length,
        lambda: flat_new_output + flat_new_state,
        # else copy some of it through
        lambda: _copy_some_through(flat_new_output, flat_new_state))

  # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
  # but benefits from removing cond() and its gradient.  We should
  # profile with and without this switch here.
  if skip_conditionals:
    # Instead of using conditionals, perform the selective copy at all time
    # steps.  This is faster when max_seq_len is equal to the number of unrolls
    # (which is typical for dynamic_rnn).
    new_output, new_state = call_cell()
    nest.assert_same_structure(zero_output, new_output)
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)
  else:
    empty_update = lambda: flat_zero_output + flat_state
    final_output_and_state = control_flow_ops.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length,
        empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

  if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
    raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
  final_output = final_output_and_state[:len(flat_zero_output)]
  final_state = final_output_and_state[len(flat_zero_output):]

  for output, flat_output in zip(final_output, flat_zero_output):
    output.set_shape(flat_output.get_shape())
  for substate, flat_substate in zip(final_state, flat_state):
    if not isinstance(substate, tensor_array_ops.TensorArray):
      substate.set_shape(flat_substate.get_shape())

  final_output = nest.pack_sequence_as(
      structure=zero_output, flat_sequence=final_output)
  final_state = nest.pack_sequence_as(
      structure=state, flat_sequence=final_state)

  return final_output, final_state