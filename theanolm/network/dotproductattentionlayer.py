#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Additive Attention layer.
"""
import logging
import theano
import theano.tensor as tensor
from theanolm.network.weightfunctions import get_submatrix
from theanolm.network.basiclayer import BasicLayer
import math
class AttentionLayer(BasicLayer):
    """
    """

    def __init__(self, layer_options, *args, **kwargs):
        """Initializes the parameters used by this layer.

        """

        super().__init__(layer_options, *args, **kwargs)

        input_size = sum(x.output_size for x in self._input_layers)
        output_size = self.input_size
        attention_size = self.output_size
        logging.debug('Attention sizes: %d %d', input_size, attention_size)
        # Attention values for time steps.
        #self.attention_value = self._network.add_recurrent_state(output_size)
        if (not 'subtype' in layer_options) or layer_options['subtype'] == 'additive':
            # layer input weights for each gate and the candidate state
            self._init_weight('layer_input/W', (input_size, attention_size), scale=0.01)
            self._init_bias('layer_input/b', attention_size)
            self.subtype = 'additive'
            logging.debug('Attention type: additive')
        elif layer_options['subtype']  == 'dotproduct':
            self._init_weight('layer_input/K', (input_size, attention_size), scale=0.01)
            self._init_weight('layer_input/Q', (input_size, attention_size), scale=0.01)
            self.subtype = 'dotproduct'
            logging.debug('Attention type: dotproduct')
        else:
            raise ValueError("Invalid subtype for attention layer")
        # final matrix to calculate attention
        self._init_weight('layer_attention/V', (attention_size, 1), scale=0.01)
        # biases for each gate and the candidate state
        #self._init_bias('layer_input/b', output_size)
        
        self.output = None
    
    def additive_attention(self, x_input):
        layer_input_preact = self._tensor_preact(x_input, 'layer_input')
        attention_weights = self._get_param('layer_attention/V')
        attention = tensor.dot(tensor.tanh(layer_input_preact), attention_weights)
        attention = tensor.swapaxes(attention, 0, 1)
        attention = attention.reshape([num_sequences, num_time_steps * self.output_size]))
        return attention
       
    def dot_product_attention(self, x_input):
        K_weights = self._get_param('layer_input/K')
        Q_weights = self._get_param('layer_input/Q')
        V_weights = self._get_param('layer_attention/V')
        Q = tensor.dot(x_input, Q_weights)
        K = tensor.dot(x_input, K_weights)
        #These shoud be moved to _create_time_step ?
        scale = math.sqrt(self.input_size)
        #softmax should not be here
        attention = tensor.dot(tensor.nnet.softmax(tensor.dot(Q, K.T)/scale), V_weights) 
        return attention

    def create_structure(self):
        """Creates the symbolic graph of this layer.

        The input is always 3-dimensional: the first dimension is the time step,
        the second dimension are the sequences, and the third dimension is the
        layer input. When processing mini-batches, all dimensions can have size
        greater than one.

        """

        layer_input = tensor.concatenate([x.output for x in self._input_layers],
                                         axis=2)
        num_time_steps = layer_input.shape[0]
        num_sequences = layer_input.shape[1]
        
        # Weights of the hidden state input of each time step have to be applied
        # inside the loop.
        attention = None
        if self.subtype == 'additive':
            attention = self.additive_attention(layer_input)
        elif self.subtype  == 'dotproduct':
            attention = self.dot_product_attention(layer_input)
        #apply attention to calculate the output vector
        #to do this we need to use theano.scan
        #the function _create_time_step will handle the time axis
        #TODO
        self.output = tensor.dot(attention, layer_input)
        
