# -*- coding: utf-8 -*-
"""A module that implements the LSTM layer with attention.
"""

import theano
import theano.tensor as tensor
import logging
from theanolm.network.weightfunctions import get_submatrix
from theanolm.network.basiclayer import BasicLayer

class LSTMXLLayer(BasicLayer):
    """Long Short-Term Memory Layer, where the long term memory is enchanced with additive attention

    """

    def __init__(self, layer_options, *args, **kwargs):
        """Initializes the parameters used by this layer.

        The weight matrices are concatenated so that they can be applied in a
        single parallel matrix operation. The same thing for bias vectors.
        Input, forget, and output gate biases are initialized to -1.0, 1.0, and
        -1.0 respectively, so that in the beginning of training, the forget gate
        activation will be almost 1.0 (meaning that the LSTM does not default to
        forgetting information).
        """

        super().__init__(layer_options, *args, **kwargs)

        input_size = sum(x.output_size for x in self._input_layers)
        output_size = self.output_size
        if 'memory_size' in layer_options:
            self.memory_size =  int(layer_options['memory_size'])
        else:
            self.memory_size = 10

        # Add state variables to be passed between time steps.
        self.cell_state_index = self._network.add_recurrent_state(output_size*self.memory_size)
        self.hidden_state_index = self._network.add_recurrent_state(output_size)

        # Initialize the parameters.
        num_gates = 3
        # layer input weights for each gate and the candidate state
        self._init_weight('layer_input/W', (input_size, output_size),
                          scale=0.01, count=num_gates+1)
        # hidden state input weights for each gate and the candidate state
        self._init_weight('step_input/W', (output_size, output_size),
                          count=num_gates+1,
                          split_to_devices=False)
        # biases for each gate and the candidate state
        self._init_bias('layer_input/b', output_size, [-1.0, 1.0, -1.0, 0.0])


        # attention parameters
        # layer weights for the output
        self._init_weight('layer_output/W', (output_size, output_size), scale=0.01)
        # biases for the transformation
        self._init_bias('layer_output/b', output_size)
        # weights for the state vector
        self._init_weight('attention/Q', (output_size, output_size), scale=0.01)
        # Value weights, to calculate the raw attention values for each timestep
        self._init_weight('attention/V', (output_size, 1), scale=0.01)
        # biases for the transformations
        self._init_bias('attention/b', 1)
        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        layer input. When processing mini-batches, all dimensions can have size
        greater than one.

        The function can also be used to create a structure for generating the
        probability distribution of the next word. Then the input is still
        3-dimensional, but the size of the first dimension (time steps) is 1,
        and the state outputs from the previous time step are read from
        ``self._network.recurrent_state_input``.

        Saves the recurrent state in the Network object: cell state C_(t) and
        hidden state h_(t). ``self.output`` will be set to the hidden state
        output, which is the actual output of this layer.
        """

        layer_input = tensor.concatenate([x.output for x in self._input_layers],
                                         axis=2)
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]

        # Compute the gate and candidate state pre-activations, which don't
        # depend on the state input from the previous time step.
        layer_input_preact = self._tensor_preact(layer_input, 'layer_input')
        if self._reverse_time:
            # Shift the input left by two time steps in the backward layer of a
            # bidirectional layer. The target word is the next input word, so
            # otherwise predicting it will be trivial.
            preact_dim = layer_input_preact.shape[2]
            padding = tensor.zeros([2, num_sequences, preact_dim])
            layer_input_preact = tensor.concatenate(
                [layer_input_preact[2:], padding],
                axis=0)

        # Weights of the hidden state input of each time step have to be applied
        # inside the loop.
        hidden_state_weights = self._get_param('step_input/W')
        mem_att_weights = self._get_param('layer_output/W')
        mem_att_bias = self._get_param('layer_output/b')
        value_att_weights = self._get_param('attention/V')
        value_att_bias = self._get_param('attention/b')
        q_att_weights = self._get_param('attention/Q')
        if self._network.mode.minibatch:
            sequences = [self._network.mask, layer_input_preact]
            non_sequences = [hidden_state_weights, mem_att_weights, mem_att_bias, value_att_weights, value_att_bias, q_att_weights]
            initial_cell_state = tensor.zeros(
                (num_sequences, self.output_size*self.memory_size), dtype=theano.config.floatX)
            initial_hidden_state = tensor.zeros(
                (num_sequences, self.output_size), dtype=theano.config.floatX)

            state_outputs, _ = theano.scan(
                self._create_time_step,
                sequences=sequences,
                outputs_info=[initial_cell_state, initial_hidden_state],
                non_sequences=non_sequences,
                go_backwards=self._reverse_time,
                name='lstm_steps',
                n_steps=num_time_steps,
                profile=self._profile,
                strict=True)

            self.output = state_outputs[1]
            if self._reverse_time:
                self.output = self.output[::-1]
        elif self._reverse_time:
            raise RuntimeError("Text generation and lattice decoding are not "
                               "possible with bidirectional layers.")
        else:
            cell_state_input = \
                self._network.recurrent_state_input[self.cell_state_index]
            hidden_state_input = \
                self._network.recurrent_state_input[self.hidden_state_index]
            #if network has attention we need to calculate all outputs, not just the current one
            state_outputs = None
            if self._network.has_attention:
                #TODO update
                sequences = [self._network.mask, layer_input_preact]
                non_sequences = [hidden_state_weights, mem_att_weights, mem_att_bias, value_att_weights, value_att_bias, q_att_weights]
                initial_cell_state = tensor.zeros(
                    (num_sequences, self.output_size*self.memory_size), dtype=theano.config.floatX)
                initial_hidden_state = tensor.zeros(
                    (num_sequences, self.output_size), dtype=theano.config.floatX)

                state_outputs, _ = theano.scan(
                    self._create_time_step,
                    sequences=sequences,
                    outputs_info=[initial_cell_state, initial_hidden_state],
                    non_sequences=non_sequences,
                    go_backwards=self._reverse_time,
                    name='lstm_steps',
                    n_steps=num_time_steps,
                    profile=self._profile,
                    strict=True)
                state_outputs[0] = state_outputs[0][-1,:,:]
                state_outputs[1] = state_outputs[1][-1,:,:]
            else:
                state_outputs = self._create_time_step(
                    self._network.mask[0],
                    layer_input_preact[0],
                    cell_state_input[0],
                    hidden_state_input[0],
                    hidden_state_weights, mem_att_weights, mem_att_bias, value_att_weights, value_att_bias, q_att_weights)

            cell_state_output = state_outputs[0]
            hidden_state_output = state_outputs[1]
            # Create a new axis for time step with size 1.
            cell_state_output = cell_state_output[None, :, :]
            hidden_state_output = hidden_state_output[None, :, :]
            self._network.recurrent_state_output[self.cell_state_index] = \
                cell_state_output
            self._network.recurrent_state_output[self.hidden_state_index] = \
                hidden_state_output
            self.output = hidden_state_output

    def _create_time_step(self, mask, x_preact, C_in, h_in, h_weights, mem_weights, mem_bias, v_weights, v_bias, q_weights):
        """The LSTM step function for theano.scan(). Creates the structure of
        one time step.

        The inputs do not contain the time step dimension. ``mask`` is a vector
        containing a boolean mask for each sequence. ``x_preact`` is a matrix
        containing the preactivations for each sequence. ``C_in`` and ``h_in``,
        as well as the outputs, are matrices containing the state vectors for
        each sequence.

        The required affine transformations have already been applied to the
        input prior to creating the loop. The transformed inputs and the mask
        that will be passed to the step function are vectors when processing a
        mini-batch - each value corresponds to the same time step in a different
        sequence.

        :type mask: Variable
        :param mask: a symbolic vector that masks out sequences that are past
                     the last word

        :type x_preact: Variable
        :param x_preact: concatenation of the input x_(t) pre-activations
                         computed using the gate and candidate state weights and
                         biases; shape is (the number of sequences, state size *
                         4)

        :type C_in: Variable
        :param C_in: C_(t-1...t-n), memory (cell output) of the previous time steps; shape
                     is (the number of sequences, state size* memory size)

        :type h_in: Variable
        :param h_in: h_(t-1), hidden state output of the previous time step;
                     shape is (the number of sequences, state size)

        :type h_weights: Variable
        :param h_weights: concatenation of the gate and candidate state weights
                          to be applied to h_(t-1); shape is (state size, state
                          size * 4)

        :rtype: a tuple of two Variables
        :returns: C_(t) and h_(t), the cell state and hidden state outputs
        """

        # pre-activation of the gates and candidate state
        preact = tensor.dot(h_in, h_weights)
        preact += x_preact
        num_sequences = x_preact.shape[0]
        # input, forget, and output gates
        i = tensor.nnet.sigmoid(get_submatrix(preact, 0, self.output_size))
        f = tensor.nnet.sigmoid(get_submatrix(preact, 1, self.output_size))
        o = tensor.nnet.sigmoid(get_submatrix(preact, 2, self.output_size))

        # hidden state outputs candidate
        h_candidate = tensor.tanh(get_submatrix(preact, 3, self.output_size))

        # calculate the attention weights
        # transforming the memory
        # First rehape C_in
        mem = C_in.reshape([num_sequences, self.memory_size, self.output_size])		
        hidden = tensor.dot(mem[:,:-1,:], mem_weights) + mem_bias
        hidden_q = (tensor.dot(h_in, q_weights)).reshape([num_sequences, 1, self.output_size])
        # use V to calculate the attention scores for all previous input vectors
        raw_attention = tensor.dot(tensor.tanh(hidden+hidden_q), v_weights) + v_bias

        #logging.debug("time: %s, seq: &s", t, num_sequences)
        raw_attention = tensor.swapaxes(raw_attention, 0, 1)
        raw_attention = raw_attention.reshape([num_sequences, self.memory_size-1])

        # with softmax we get the attention scores for each time t
        # shape is (num_sequences, t)
        attentions = tensor.nnet.softmax(raw_attention)
        # apply attention to the memory
        long_memory = tensor.batched_dot(attentions.reshape([attentions.shape[0],1,attentions.shape[1]]), mem[:,:-1,:]) #TODO test
        long_memory = long_memory.reshape([long_memory.shape[0], long_memory.shape[2]])
        h_out = o * self._activation(f * long_memory + i * h_candidate)
        #concat new vector
        logging.debug("C ndim: %s, h_out ndim: %s", C_in.ndim, h_out.ndim)
        mem = tensor.concatenate([C_in[:,self.output_size:], h_out], axis=1) # TODO chech dimensions!

        # Apply the mask. None creates a new axis with size 1, causing the mask
        # to be broadcast to all the outputs.
        #C_out = tensor.switch(mask[:, None], C_out, C_in)
        h_out = tensor.switch(mask[:, None], h_out, h_in)

        return mem, h_out

