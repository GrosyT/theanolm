#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Attention layer.
"""

import theano
import theano.tensor as tensor
import logging
from theanolm.network.weightfunctions import get_submatrix
from theanolm.network.basiclayer import BasicLayer

class AdditiveAttentionLayer(BasicLayer):
    """Additive Attention Layer

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015)
    Neural Machine Translation by Jointly Learning to Align and Translate
    ICLR
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Initializes the parameters used by this layer.
        """

        super().__init__(layer_options, *args, **kwargs)

        input_size = sum(x.output_size for x in self._input_layers)
        output_size = self.output_size
        if input_size != output_size:
            raise ValueError("The input and output size for attention layers must be equal!")
        if 'hidden_size' in layer_options:
            self.hidden_size = int(layer_options['hidden_size'])
        else:
            self.hidden_size = output_size
        hidden_size = self.hidden_size
        logging.debug("Attention layer hidden dimension: %d", hidden_size)
        # Initialize the parameters.
        
        # layer weights for the input
        self._init_weight('layer_input/W', (input_size, hidden_size), scale=0.01)
        # layer weights for the output
        self._init_weight('layer_output/W2', (output_size, hidden_size), scale=0.01)
        # biases for the transformations
        self._init_bias('layer_input/b', hidden_size)
        self._init_bias('layer_output/b', hidden_size)
        
        # Value weigths, to calculate the raw attention values for each timestep
        self._init_weight('attention/V', (hidden_size, 1), scale=0.01)
        # biases for the transformations
        self._init_bias('attention/b', 1)
        
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
        3-dimensional, but the size of the first dimension (time steps) is 1.
        """

        layer_input = tensor.concatenate([x.output for x in self._input_layers],
                                         axis=2)
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]

        # Compute the gate and candidate state pre-activations, which don't
        # depend on the state input from the previous time step.
        layer_input_preact = self._tensor_preact(layer_input, 'layer_input')

        # Weights of the hidden state input and value layer 
        # of each time step have to be applied inside the loop.
        hidden_state_weights = self._get_param('layer_output/W2')
        hidden_state_bias = self._get_param('layer_output/b')
        value_state_weights = self._get_param('attention/V')
        value_state_bias = self._get_param('attention/b')   
        if self._network.mode.minibatch:
            sequences = [tensor.arange(num_time_steps), self._network.mask]
            non_sequences = [layer_input, layer_input_preact, hidden_state_weights, hidden_state_bias, value_state_weights, value_state_bias]
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
                layer_input,
                layer_input_preact,
                hidden_state_weights, hidden_state_bias, 
                value_state_weights, value_state_bias)
            hidden_state_output = state_outputs

            # Create a new axis for time step with size 1.
            hidden_state_output = hidden_state_output[None, :, :]

            self._network.recurrent_state_output[self.hidden_state_index] = \
                hidden_state_output
            self.output = hidden_state_output

    def _calc_sum(self, location, x, attentions):
        """Multiply the input vectors with the attention weights and sum them along the time axis
        """
        return tensor.tensordot(x[:,location,:],attentions[location,:], [[0],[0]])

    def _create_time_step(self, t, mask, S, x, x_preact, h_weights, h_bias, v_weights, v_bias):
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
        
        :type x: Variable
        :param x: concatenation of the input vectors x_(t)

        :type x_preact: Variable
        :param x_preact: concatenation of the input x_(t) pre-activations
                         computed using the W1 weights and
                         biases; shape is (the number of sequences, num_hiden)

        :type S: Variable
        :param S: C_(t-1), layer output of the previous time step; shape
                     is (the number of sequences, state size)

        :type h_weights: Variable
        :param h_weights: weights to be applied to S_(t-1); 
                     shape is (output_size, hidden_size)

        :type h_bias: Variable
        :param h_bias: bias to be applied with h_weights

        :type v_weights: Variable
        :param v_weights: value weights to calculate 
                     the raw attention score; shape is (hidden_size, 1)

        :type v_bias: Variable
        :param v_bias: weights to be applied with v_weights

        :rtype: matrix of output
        :returns: attended context vector for timestep t
        """        
        if t == 0:
            return x[0,:,:]
        # transforming the previous output
        hidden = tensor.dot(S, h_weights) + h_bias

        # use V to calculate the attention scores for all previous input vectors
        raw_attention = tensor.dot(tensor.tanh(x_preact[:t+1,:,:]+hidden), v_weights) + v_bias

        # Apply softmax to calculate the current attention weights
        # first reshape the 3D tensor into a 2D one
        num_sequences = x.shape[1]
        #logging.debug("time: %s, seq: &s", t, num_sequences)
        raw_attention = tensor.swapaxes(raw_attention, 0, 1)
        raw_attention = raw_attention.reshape([num_sequences, raw_attention.shape[1]])
        
        # with softmax we get the attention scores for each time t
        # shape is (num_sequences, t)               
        attentions = tensor.nnet.softmax(raw_attention)
   
        # Calculate the new output using the attention weights
        # multiply the input vectors with the appropiate attention score
        C_out = tensor.batched_dot(attentions.reshape([attentions.shape[0],1,attentions.shape[1]]), x[:t+1, :, :].dimshuffle((1,0,2)))
        C_out = C_out.reshape([C_out.shape[0], C_out.shape[2]])
        #non_seq = [x[:t+1,:,:], attentions]
        #location = tensor.arange(num_sequences)
        #C_out,_ = theano.scan(fn=self._calc_sum,
        #                      outputs_info=None,
        #                      sequences=[location],
        #                      non_sequences=non_seq)
        # Apply the mask. None creates a new axis with size 1, causing the mask
        # to be broadcast to all the outputs.
        C_out = tensor.switch(mask[:, None], C_out, S)

        return C_out
