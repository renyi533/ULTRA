from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf

import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils
from ultra.utils import losses_impl
from ultra.utils import utils_func
import ultra.utils as utils

DEFAULT_GAIN_FN = lambda label: label
EXP_GAIN_FN = lambda label: tf.pow(2.0, label) - 1
DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)
RECIPROCAL_RANK_DISCOUNT_FN = lambda rank: 1. / rank

def create_dcg_lambda_weight(discount_func, gain_func, topn=None, normalized=True, smooth_fraction=0.):
    """Creates _LambdaWeight for DCG metric."""
    RANK_DISCOUNT_FN = DEFAULT_RANK_DISCOUNT_FN if discount_func == 'log1p' else RECIPROCAL_RANK_DISCOUNT_FN
    GAIN_FN = DEFAULT_GAIN_FN if gain_func == 'linear' else EXP_GAIN_FN
    return losses_impl.DCGLambdaWeight(
        topn,
        gain_fn=GAIN_FN,
        rank_discount_fn=RANK_DISCOUNT_FN,
        normalized=normalized,
        smooth_fraction=smooth_fraction)

class PairwiseAdversarial(BaseAlgorithm):
    """The Pairwise Adversarial algorithm for unbiased learning to rank.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Pairwise Adversarial algorithm.')
        self.tau = 1e-12
        self.hparams = ultra.utils.hparams.HParams(
            adv_learning_rate=0.05,         # Learning rate for adversarial model
            learning_rate=0.05,             # Learning rate.
            step_decay_ratio=-0.0,
            alpha=5.0,
            max_gradient_norm=5.0,          # Clip gradients to this norm.
            gain_fn='exp',
            discount_fn='log1p',
            opt_metric='ndcg',
            mode='adv',
            naive_loss='pairwise',
            # Set strength for L2 regularization.
            l2_loss=0.00,
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        print('hparams:')
        print(self.hparams)
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False) * \
            tf.math.pow((1+tf.cast(self.global_step, tf.float32)), \
                        self.hparams.step_decay_ratio)
        self.adv_learning_rate = self.hparams.adv_learning_rate * \
            tf.math.pow((1+tf.cast(self.global_step, tf.float32)), \
                        self.hparams.step_decay_ratio)
     
        self.lambda_weight = None
        if self.hparams.opt_metric == 'ndcg':
            self.lambda_weight = create_dcg_lambda_weight(self.hparams.discount_fn, \
                                                          self.hparams.gain_fn, \
                                                          normalized=True)
        elif self.hparams.opt_metric == 'dcg':
            self.lambda_weight = create_dcg_lambda_weight(self.hparams.discount_fn, \
                                                          self.hparams.gain_fn, \
                                                          normalized=False)
        print(self.lambda_weight)
        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                    name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                              name="label{0}".format(i)))
        extra_model_cfg = ',shared_layers=2,target_cnt=2'
        self.adv_model = utils.find_class(
                    'ultra.ranking_model.SharedBottomDNN')(
                    self.exp_settings['ranking_model_hparams']+extra_model_cfg)
        assert self.hparams.mode == 'adv' or self.hparams.mode == 'bias_corr' \
            or self.hparams.mode == 'ips' or self.hparams.mode == 'naive'
        if self.hparams.mode != 'bias_corr':
            self.output = self.ranking_model(
                self.max_candidate_num, scope='ips_ranking_model')
        else:
            self.output = self.ranking_model(
                self.max_candidate_num, scope='naive_ranking_model', model=self.adv_model)
            self.output, _ = tf.split(self.output, 2, axis=1)
        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels))
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                tf.summary.scalar(
                    '%s_%d' %
                    (metric, topn), metric_value, collections=['eval'])

        # Build unbiased pairwise loss only when it is training
        if not forward_only:
            self.rank_list_size = exp_settings['selection_bias_cutoff']

            train_output = self.ranking_model(
                self.rank_list_size, scope='naive_ranking_model', model=self.adv_model)
            train_output, bias_output = tf.split(train_output, 2, axis=1)
            self.position_bias = self.DenoisingNet(
                self.rank_list_size, forward_only, scope='naive_ranking_model')  
            #train_output += self.position_bias 
            exposure_prob = tf.math.sigmoid(self.position_bias)
            prob = tf.math.sigmoid(train_output) * exposure_prob
            comb_train_output = tf.math.log(prob/(1-prob))
            ips_train_output = self.ranking_model(
                self.rank_list_size, scope='ips_ranking_model')
            train_labels = self.labels[:self.rank_list_size]
            tf.summary.histogram("ips_train_output", ips_train_output, 
                collections=['train'])
            tf.summary.histogram("comb_train_output", comb_train_output, 
                collections=['train'])
            tf.summary.histogram("naive_train_output", train_output, 
                collections=['train'])
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            tf.summary.histogram("reshaped_train_labels", reshaped_train_labels, 
                collections=['train'])
            binary_labels = tf.where(tf.math.greater(reshaped_train_labels,0.0), \
                x=tf.ones_like(reshaped_train_labels, dtype=tf.float32), \
                y=tf.zeros_like(reshaped_train_labels, dtype=tf.float32))
            tf.summary.histogram("binary_labels", binary_labels, 
                collections=['train'])
            self.construct_loss(comb_train_output, ips_train_output, \
                reshaped_train_labels, binary_labels)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                                    labels = tf.stop_gradient(exposure_prob),
                                    logits = bias_output
                                )
            self.adv_loss = tf.reduce_mean(
                                    tf.reduce_sum(
                                        losses,
                                        axis=[1]
                                    )
                                )
            tf.summary.scalar(
                'adv_loss', tf.reduce_mean(
                    self.adv_loss), collections=['train'])
            adv_expose_prob = tf.math.sigmoid(bias_output)
            max_prob = tf.reduce_max(adv_expose_prob, axis=[1], keepdims=True)
            adv_expose_prob = adv_expose_prob / max_prob
            self.debug_vector_tensor(adv_expose_prob, 'adv_expose_prob', self.rank_list_size)
            if self.hparams.mode == 'adv':
                self.loss = self.loss - self.hparams.alpha * self.adv_loss

            # Add l2 loss
            params = tf.trainable_variables()
            print('trainable vars:')
            print(params)	    
            if self.hparams.l2_loss > 0:
                for p in params:
                    if p.op.name.find('naive_ranking_model/dnn_1_layers') == -1:
                        self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
                    else:
                        self.loss -= self.hparams.l2_loss * tf.nn.l2_loss(p)

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            # Gradients and SGD update operation for training the model.
            opt = self.optimizer_func(self.learning_rate)
            adv_opt = self.optimizer_func(self.adv_learning_rate)
            self.gradients = tf.gradients(self.loss, params)

            #for var in tf.global_variables():
            #    tf.summary.histogram(var.op.name, var)
            adv_grad = []
            adv_var = []
            other_grad = []
            other_var = []
            for grad,var in zip(self.gradients, params):
                tf.summary.histogram(var.op.name, var, collections=['train'])
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', 
                            grad, collections=['train'])
                    if var.op.name.find('naive_ranking_model/dnn_1_layers') == -1:
                        other_grad.append(grad)
                        other_var.append(var)
                    else:
                        adv_grad.append(-grad / self.hparams.alpha)
                        adv_var.append(var)

            self.norm = tf.global_norm(other_grad)
            self.adv_norm = tf.global_norm(adv_grad)

            tf.summary.scalar(
                'Gradient_Norm',
                self.norm,
                collections=['train'])
            tf.summary.scalar(
                'Adv_Gradient_Norm',
                self.adv_norm,
                collections=['train'])

            print('adversarial vars:')
            print(adv_var)
            print('other vars:')
            print(other_var)
            if self.hparams.max_gradient_norm > 0:
                print('clip gradients separately')
                other_grad, _ = tf.clip_by_global_norm(other_grad,
                                                       self.hparams.max_gradient_norm,
                                                       use_norm=self.norm)
                adv_grad, _ = tf.clip_by_global_norm(adv_grad,
                                                     self.hparams.max_gradient_norm,
                                                     use_norm=self.adv_norm)
                tf.summary.scalar(
                    'Gradient_Norm_after_clip',
                    tf.global_norm(other_grad),
                    collections=['train'])
                tf.summary.scalar(
                    'Adv_Gradient_Norm_after_clip',
                    tf.global_norm(adv_grad),
                    collections=['train'])


            self.updates = opt.apply_gradients(zip(other_grad, other_var),
                                               global_step=self.global_step)
            self.adv_updates = adv_opt.apply_gradients(zip(adv_grad, adv_var))
            tf.summary.scalar(
                'Learning_Rate',
                self.learning_rate,
                collections=['train'])
            tf.summary.scalar(
                'adv_learning_rate',
                self.adv_learning_rate,
                collections=['train'])   
            tf.summary.scalar(
                'Loss', tf.reduce_mean(
                    self.loss), collections=['train'])

            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            pad_removed_train_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, train_output)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_train_output, None)
                    tf.summary.scalar(
                        '%s_%d' %
                        (metric, topn), metric_value, collections=['train'])

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def construct_loss(self, train_output, ips_train_output, \
            train_labels, binary_labels):
        # Conduct pairwise expectation step
        train_labels_j = tf.expand_dims(train_labels, axis=1)
        train_labels_i = tf.expand_dims(train_labels, axis=-1)
        binary_labels_j = tf.expand_dims(binary_labels, axis=1)
        binary_labels_i = tf.expand_dims(binary_labels, axis=-1)

        delta_labels = train_labels_i - train_labels_j
        tf.summary.histogram("pairwise_delta_labels", delta_labels, 
                collections=['train'])
        pairwise_labels = tf.where(tf.math.greater(delta_labels,0.0), \
            x=tf.ones_like(delta_labels, dtype=tf.float32), \
            y=tf.zeros_like(delta_labels, dtype=tf.float32))
        pairwise_binary_labels = tf.nn.relu(binary_labels_i - binary_labels_j)
        tf.summary.histogram("pairwise_labels", pairwise_labels, 
                collections=['train'])
        tf.summary.histogram("pairwise_binary_labels", pairwise_binary_labels, 
                collections=['train'])

        train_output_j = tf.expand_dims(train_output, axis=1)
        train_output_i = tf.expand_dims(train_output, axis=-1)
        pairwise_logits = train_output_i - train_output_j
        tf.summary.histogram("pairwise_logits", 
                pairwise_logits*pairwise_labels, 
                collections=['train'])

        mask = utils_func.is_label_valid(train_labels)
        ranks = losses_impl._compute_ranks(train_output, mask)

        if self.lambda_weight is not None:
            pairwise_weights = self.lambda_weight.pair_weights(train_labels, ranks)
            pairwise_weights *= tf.cast(tf.shape(input=train_labels)[1], dtype=tf.float32)

            pairwise_weights = tf.stop_gradient(
                pairwise_weights, name='weights_stop_gradient')
        else:
            pairwise_weights = tf.ones([], dtype=tf.float32)

        tf.summary.histogram("pairwise_weights", 
                pairwise_weights*pairwise_labels, 
                collections=['train'])

        assert self.hparams.naive_loss == 'pairwise' or self.hparams.naive_loss == 'pointwise'
        if self.hparams.naive_loss == 'pairwise':
            losses = pairwise_binary_labels * tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=pairwise_binary_labels, logits=pairwise_logits)
            self.naive_loss = tf.reduce_mean(
                                    tf.reduce_sum(
                                        losses,
                                        axis=[1,2]
                                    )
                                )
        else:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_labels, logits=train_output)      
            self.naive_loss = tf.reduce_mean(
                                    tf.reduce_sum(
                                        losses,
                                        axis=[1]
                                    )
                                )

        tf.summary.scalar(
            'naive_loss', tf.reduce_mean(
                self.naive_loss), collections=['train'])

        ips_train_output_j = tf.expand_dims(ips_train_output, axis=1)
        ips_train_output_i = tf.expand_dims(ips_train_output, axis=-1)
        ips_pairwise_logits = ips_train_output_i - ips_train_output_j
        tf.summary.histogram("ips_pairwise_logits", 
                ips_pairwise_logits*pairwise_labels, 
                collections=['train'])
        
        position_bias_i = tf.expand_dims(self.position_bias, axis=-1)
        print('mode: %s' % self.hparams.mode)
        if self.hparams.mode == 'naive' or self.hparams.mode == 'bias_corr':
            logits = 100.0 * tf.ones_like(ips_pairwise_logits)
        else:
            logits = position_bias_i 

        expose_prob = tf.math.sigmoid(logits)
        max_prob = tf.reduce_max(expose_prob, axis=[1,2], keepdims=True)
        expose_prob = expose_prob / max_prob
        expose_prob_mean = tf.reduce_mean(expose_prob, axis=[2], keepdims=False)
        self.debug_vector_tensor(expose_prob_mean, 'expose_prob_mean', self.rank_list_size)
        ips_weights = 1 / (expose_prob + self.tau)

        ips_weights = tf.stop_gradient(ips_weights)

        ips_weights_mean = tf.reduce_mean(ips_weights, axis=[2], keepdims=False)
        self.debug_vector_tensor(ips_weights_mean, 'ips_weights_mean', self.rank_list_size)

        tf.summary.histogram("ips_weights", ips_weights * pairwise_labels, 
                collections=['train'])

        losses = pairwise_labels * tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=pairwise_labels, logits=ips_pairwise_logits)
        losses = losses * pairwise_weights * ips_weights
        self.ips_loss = tf.reduce_mean(
                                    tf.reduce_sum(
                                        losses,
                                        axis=[1,2]
                                    )
                                )
        tf.summary.scalar(
            'ips_loss', tf.reduce_mean(
                self.ips_loss), collections=['train'])
        self.loss = self.ips_loss + self.naive_loss

    def DenoisingNet(self, list_size, forward_only=False, scope=None):
        with tf.variable_scope(scope or "denoising_model"):
            # If we are in testing, do not compute propensity
            if forward_only:
                return tf.ones_like(self.output)  # , tf.ones_like(self.output)
            input_vec_size = list_size

            def propensity_network(input_data, index):
                reuse = None if index < 1 else True
                propensity_initializer = None
                with tf.variable_scope("propensity_network", initializer=propensity_initializer,
                                       reuse=reuse):
                    output_data = input_data
                    current_size = input_vec_size
                    output_sizes = [
                        #int((list_size+1)/2) + 1,
                        #int((list_size+1)/4) + 1,
                        1
                    ]
                    for i in range(len(output_sizes)):
                        expand_W = tf.get_variable(
                            "W_%d" % i, [current_size, output_sizes[i]])
                        expand_b = tf.get_variable(
                            "b_%d" % i, [output_sizes[i]])
                        output_data = tf.nn.bias_add(
                            tf.matmul(output_data, expand_W), expand_b)
                        if i != len(output_sizes) - 1:
                            output_data = tf.nn.elu(output_data)
                        current_size = output_sizes[i]
                    #expand_W = tf.get_variable("final_W", [current_size, 1])
                    #expand_b = tf.get_variable("final_b" , [1])
                    #output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
                    return output_data

            output_propensity_list = []
            for i in range(list_size):
                # Add position information (one-hot vector)
                click_feature = [
                    tf.expand_dims(
                        tf.zeros_like(
                            self.labels[i]), -1) for _ in range(list_size)]
                click_feature[i] = tf.expand_dims(
                    tf.ones_like(self.labels[i]), -1)
                # Predict propensity with a simple network
                output_propensity_list.append(
                    propensity_network(
                        tf.concat(
                            click_feature, 1), i))

        return tf.concat(output_propensity_list, 1)

    def debug_vector_tensor(self, var, name, size):
        splitted_var = tf.split(var, size, axis=1)
        print('debug vector tensor: ' + name)

        for i in range(size):
            tf.summary.scalar(
                    '%s_%d' % (name, i),
                    tf.reduce_mean(
                        splitted_var[i]),
                    collections=['train'])
            tf.summary.histogram(
                            '%s_%d_histogram' % (name, i),
                            splitted_var[i],
                            collections=['train'])

    def step(self, session, input_feed, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            session: (tf.Session) tensorflow session to use.
            input_feed: (dictionary) A dictionary containing all the input feed data.
            forward_only: whether to do the backward step (False) or only forward (True).

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.adv_updates,
                self.train_summary  # Summarize statistics.
            ]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output   # Model outputs
            ]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # loss, no outputs, summary.
            return outputs[1], None, outputs[-1]
        else:
            return None, outputs[1], outputs[0]    # loss, outputs, summary.

