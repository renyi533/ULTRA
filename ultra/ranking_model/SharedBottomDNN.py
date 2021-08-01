from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from ultra.ranking_model import BaseRankingModel
from ultra.ranking_model import ActivationFunctions
from ultra.ranking_model import NormalizationFunctions
import ultra.utils

class SharedBottomDNN(BaseRankingModel):
    """The deep neural network model for learning to rank.

    This class implements a deep neural network (DNN) based ranking model with bias feature. It's essientially a multi-layer perceptron network.

    """
    def __init__(self, hparams_str):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = ultra.utils.hparams.HParams(
            # Number of neurons in each layer of a ranking_model.
            hidden_layer_sizes=[512, 256, 128],
            # Type for activation function, which could be elu, relu, sigmoid,
            # or tanh
            shared_layers=2,
            target_cnt=2,
            activation_func='elu',
            initializer='glorot',                         # Set parameter initializer
            norm="layer"                                # Set the default normalization
        )
        self.hparams.parse(hparams_str)
        self.initializer = None
        self.act_func = None
        self.layer_norm = {}
        self.shared_layer_norm = {}

        if self.hparams.activation_func in BaseRankingModel.ACT_FUNC_DIC:
            self.act_func = BaseRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]

        if self.hparams.initializer in BaseRankingModel.INITIALIZER_DIC:
            self.initializer = BaseRankingModel.INITIALIZER_DIC[self.hparams.initializer]

        self.model_parameters = {}

    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the DNN model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """

        with tf.variable_scope(tf.get_variable_scope(), initializer=self.initializer,
                               reuse=tf.AUTO_REUSE):
            var_scope_name = tf.get_variable_scope().name
            input_data = tf.concat(input_list, axis=0)
            output_data = input_data
            output_sizes = self.hparams.hidden_layer_sizes + [1]
            if (var_scope_name not in self.layer_norm) \
                and (self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC):

                self.layer_norm[var_scope_name] = [[]] * self.hparams.target_cnt
                self.shared_layer_norm[var_scope_name] = []
                for j in range(self.hparams.shared_layers):
                    self.shared_layer_norm[var_scope_name].append(BaseRankingModel.NORM_FUNC_DIC[self.hparams.norm](
                        name="shared_%s_norm_%d" % (self.hparams.norm, j)))            
                for j in range(len(output_sizes)):
                    for k in range(self.hparams.target_cnt):
                        self.layer_norm[var_scope_name][k].append(BaseRankingModel.NORM_FUNC_DIC[self.hparams.norm](
                            name="%s_norm_%d_%d" % (self.hparams.norm, k,j)))

            layer_norm = None
            shared_layer_norm = None
            if var_scope_name in self.layer_norm:
                layer_norm = self.layer_norm[var_scope_name]
                print('layer_norm objs for %s is:' % var_scope_name)
                print(layer_norm)
                shared_layer_norm = self.shared_layer_norm[var_scope_name]
                print('shared_layer_norm objs for %s is:' % var_scope_name)
                print(shared_layer_norm)
            else:
                print('no layer_norm objs for %s' % var_scope_name)

            shared_output_sizes = [self.hparams.hidden_layer_sizes[0]] * self.hparams.shared_layers
            with tf.variable_scope('shared_layers'):
                output_data = self.dnn_tower(shared_output_sizes, output_data, shared_layer_norm, 'shared_dnn', is_training)
                output_data = self.act_func(output_data)
            
            dnn_output_data = []
            for k in range(self.hparams.target_cnt):
              with tf.variable_scope('dnn_%d_layers' % k):
                dnn_output_data.append(self.dnn_tower(output_sizes, output_data, layer_norm[k], 'dnn', is_training))

            output = tf.concat(dnn_output_data, axis=0)
            return tf.split(output,  self.hparams.target_cnt * len(input_list), axis=0)
    
    def dnn_tower(self, output_sizes, input_data, layer_norm, name, is_training):
        output_data = input_data
        current_size = output_data.get_shape()[-1].value
        for j in range(len(output_sizes)):
            if layer_norm is not None:
                if self.hparams.norm == "layer":
                    output_data = layer_norm[j](
                        output_data)
                else:
                    output_data = layer_norm[j](
                        output_data, training=is_training)
            expand_W = self.get_variable(
                "%s_W_%d" % (name, j), [current_size, output_sizes[j]])
            expand_b = self.get_variable("%s_b_%d" % (name, j), [
                                         output_sizes[j]])
            output_data = tf.nn.bias_add(
                tf.matmul(output_data, expand_W), expand_b)
            # Add activation if it is a hidden layer
            if j != len(output_sizes) - 1:
                output_data = self.act_func(output_data)
            current_size = output_sizes[j]
        return output_data