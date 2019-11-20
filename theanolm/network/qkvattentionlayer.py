#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Scaled-dot Product Attention layer.
"""

import theano
import theano.tensor as tensor
import logging
from theanolm.network.weightfunctions import get_submatrix
from theanolm.network.basiclayer import BasicLayer

class QKVAttentionLayer(BasicLayer):
    """Implementation of scaled dot product attention according to:
            "Attention is all you need" by A Vaswani, N Shazeer, N Parmar (2017)
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(layer_options, *args, **kwargs)

        input_size = sum(x.output_size for x in self._input_layers)
        output_size = self.output_size


        # Initialize the parameters.
        
        # layer parameters for the query
        self._init_weight('layer_query/W', (output_size, output_size), scale=0.01)
        self._init_bias('layer_query/b', output_size)
        # layer parameters for the key
        self._init_weight('layer_key/W', (input_size, output_size), scale=0.01)
        self._init_bias('layer_key/b', output_size)
        # layer parameters for the value
        self._init_weight('layer_value/W', (input_size, output_size), scale=0.01)
        self._init_bias('layer_value/b', output_size)

        self.hidden_state_index = self._network.add_recurrent_state(output_size)
        self.output = None

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        layer input. When processing mini-batches, all dimensions can have size
        greater than one.

        The function can also be used to create a structure for generating the
        probability distribution of the next word. Then the input is still
        3-dimensional, but the size of the first dimension (time steps) is 1
        """

        layer_input = tensor.concatenate([x.output for x in self._input_layers],
                                         axis=2)
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]


        #layer_q = self._tensor_preact(layer_input, 'layer_query')
        layer_k = self._tensor_preact(layer_input, 'layer_key')
        layer_v = self._tensor_preact(layer_input, 'layer_value')
        q_weights = self._get_param('layer_query/W')
        q_bias = self._get_param('layer_query/b')

        if self._network.mode.minibatch:
            sequences = [tensor.arange(num_time_steps), self._network.mask]
            non_sequences = [layer_k, layer_v, q_weights, q_bias]
            initial_output_state = tensor.zeros(
                (num_sequences, self.output_size), dtype=theano.config.floatX)

            state_outputs, _ = theano.scan(
                self._create_time_step,
                sequences=sequences,
                outputs_info=[initial_output_state],
                non_sequences=non_sequences,
                name='attention_steps',
                n_steps=num_time_steps,
                profile=self._profile,
                strict=True)

            self.output = state_outputs
        else:
            hidden_state_input = \
                self._network.recurrent_state_input[self.hidden_state_index]
            hidden_state_input = hidden_state_input.reshape([1, self.output_size]).repeat(num_sequences, axis=0)
            state_outputs = self._create_time_step(
                num_time_steps,
                self._network.mask[0],
                hidden_state_input,
                layer_k, layer_v, q_weights, q_bias)
            hidden_state_output = state_outputs

            # Create a new axis for time step with size 1.
            hidden_state_output = hidden_state_output[None, :, :]

            self._network.recurrent_state_output[self.hidden_state_index] = \
                hidden_state_output
            self.output = hidden_state_output

    def _create_time_step(self, t, mask, S, k, v, q_w, q_b):
        """The Attention step function for theano.scan(). Creates the structure of
        one time step.

        The inputs do not contain the time step dimension. ``mask`` is a vector
        containing a boolean mask for each sequence. ``x_preact`` is a matrix
        containing the preactivations for each sequence. 

        The required affine transformations have already been applied to the
        input prior to creating the loop. The transformed inputs and the mask
        that will be passed to the step function are vectors when processing a
        mini-batch - each value corresponds to the same time step in a different
        sequence.

        :type t: Variable
        :param t: the current time index

        :type mask: Variable
        :param mask: a symbolic vector that masks out sequences that are past
                     the last word

        :rtype: matrix of output
        :returns: attended context vector for timestep t
        """        
        if t == 0:
            return v[0,:,:]
        #Calculate the Query from previous output
        q = tensor.dot(S, q_w) + q_b

        # calculate Q*K and scale it
        scale = tensor.sqrt(float(self.output_size))
        qk = tensor.batched_dot(k[:t+1, :, :].dimshuffle((1,0,2) ), q.reshape([q.shape[0],q.shape[1],1]))/scale

        # apply softmax, the shape of qk is (sequences, time, 1)
        # reshape and qk so that softmax could be applied
        qk = qk.reshape([qk.shape[0],qk.shape[1]])
        attention = tensor.nnet.softmax(qk)

        # multiply attention with V
        C_out = tensor.batched_dot(attention.reshape([attention.shape[0],1,attention.shape[1]]), v[:t+1, :, :].dimshuffle((1,0,2)))
        C_out = C_out.reshape([C_out.shape[0], C_out.shape[2]])
        C_out = tensor.switch(mask[:, None], C_out, S)

        return C_out
