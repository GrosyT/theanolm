#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy
import theano
import theano.tensor as tensor
from theanolm.optimizers.basicoptimizer import BasicOptimizer

class AdamOptimizer(BasicOptimizer):
    """Adam Optimization Method

    D.P. Kingma, J. Ba (2015)
    Adam: A Method for Stochastic Optimization
    The International Conference on Learning Representations (ICLR), San Diego
    """

    def __init__(self, optimization_options, network, *args, **kwargs):
        """Creates an Adam optimizer.

        :type optimization_options: dict
        :param optimization_options: a dictionary of optimization options

        :type network: Network
        :param network: the neural network object
        """

        self.param_init_values = dict()

        # Learning rate / step size will change during the iterations, so we'll
        # make it a shared variable.
        if not 'learning_rate' in optimization_options:
            raise ValueError("Learning rate is not given in optimization "
                             "options.")
        self.param_init_values['optimizer.learning_rate'] = \
            numpy.dtype(theano.config.floatX).type(
                optimization_options['learning_rate'])

        self.param_init_values['optimizer.timestep'] = \
            numpy.dtype(theano.config.floatX).type(0.0)

        for name, param in network.params.items():
            self.param_init_values[name + '.gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_gradient'] = \
                numpy.zeros_like(param.get_value())
            self.param_init_values[name + '.mean_sqr_gradient'] = \
                numpy.zeros_like(param.get_value())
        self._create_params()

        # geometric rate for averaging gradients
        if not 'gradient_decay_rate' in optimization_options:
            raise ValueError("Gradient decay rate is not given in training "
                             "options.")
        self._gamma_m = optimization_options['gradient_decay_rate']

        # geometric rate for averaging squared gradients
        if not 'sqr_gradient_decay_rate' in optimization_options:
            raise ValueError("Squared gradient decay rate is not given in "
                             "optimization options.")
        self._gamma_ms = optimization_options['sqr_gradient_decay_rate']

        # numerical stability / smoothing term to prevent divide-by-zero
        if not 'epsilon' in optimization_options:
            raise ValueError("Epsilon is not given in optimization options.")
        self._epsilon = optimization_options['epsilon']

        # momentum
        if not 'momentum' in optimization_options:
            raise ValueError("Momentum is not given in optimization options.")
        self._momentum = optimization_options['momentum']

        super().__init__(optimization_options, network, *args, **kwargs)

    def _get_gradient_updates(self):
        result = []
        for name, gradient_new in zip(self.network.params,
                                      self._gradient_exprs):
            gradient = self.params[name + '.gradient']
            m_gradient = self.params[name + '.mean_gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            m_gradient_new = \
                (self._gamma_m * m_gradient) + \
                ((1.0 - self._gamma_m) * gradient)
            ms_gradient_new = \
                (self._gamma_ms * ms_gradient) + \
                ((1.0 - self._gamma_ms) * tensor.sqr(gradient))
            result.append((gradient, gradient_new))
            result.append((m_gradient, m_gradient_new))
            result.append((ms_gradient, ms_gradient_new))
        return result

    def _get_model_updates(self):
        timestep = self.params['optimizer.timestep']
        timestep_new = timestep + 1.0
        alpha = self.params['optimizer.learning_rate']
        alpha *= tensor.sqrt(1.0 - (self._gamma_ms ** timestep_new))
        alpha /= 1.0 - (self._gamma_m ** timestep_new)

        result = []
        for name, param in self.network.params.items():
            m_gradient = self.params[name + '.mean_gradient']
            ms_gradient = self.params[name + '.mean_sqr_gradient']
            rms_gradient = tensor.sqrt(ms_gradient) + self._epsilon
            param_new = param - (alpha * m_gradient / rms_gradient)
            result.append((param, param_new))
        result.append((timestep, timestep_new))
        return result

    def reset(self):
        """Resets the optimizer timestep. May be called after decreasing
        learning rate, depending on the program options.
        """

        logging.info("Resetting optimizer timestep to zero.")
        self.params['optimizer.timestep'].set_value(
            numpy.dtype(theano.config.floatX).type(0.0))