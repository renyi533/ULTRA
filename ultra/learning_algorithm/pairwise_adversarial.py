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
            alpha=1.0,
            max_gradient_norm=5.0,          # Clip gradients to this norm.
            gain_fn='exp',
            discount_fn='log1p',
            opt_metric='ndcg',
            mode='adv',
            relative_corr=False,
            self_norm_ips=0,
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

        self.output = self.ranking_model(
            self.max_candidate_num, scope='ips_ranking_model')            

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
                self.rank_list_size, scope='naive_ranking_model')
            ips_train_output = self.ranking_model(
                self.rank_list_size, scope='ips_ranking_model')
            
            train_labels = self.labels[:self.rank_list_size]
            self.build_variables()
            tf.summary.histogram("ips_train_output", ips_train_output, 
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
            self.construct_loss(train_output, ips_train_output, \
                reshaped_train_labels, binary_labels)

            # Add l2 loss
            params = tf.trainable_variables()
            print('trainable vars:')
            print(params)	    
            if self.hparams.l2_loss > 0:
                for p in params:
                    self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)

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
                    if var.op.name.find('naive_ranking_model') == -1:
                        other_grad.append(grad)
                        other_var.append(var)
                    else:
                        adv_grad.append(-grad)
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

        delta_labels = train_labels_i - train_labels_j
        tf.summary.histogram("pairwise_delta_labels", delta_labels, 
                collections=['train'])
        pairwise_labels = tf.where(tf.math.greater(delta_labels,0.0), \
            x=tf.ones_like(delta_labels, dtype=tf.float32), \
            y=tf.zeros_like(delta_labels, dtype=tf.float32))
        tf.summary.histogram("pairwise_binary_labels", pairwise_labels, 
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
        losses = pairwise_labels * tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=pairwise_labels, logits=pairwise_logits)
        losses = losses * pairwise_weights          

        self.reg_loss = tf.reduce_mean(
                                    tf.reduce_sum(
                                        losses,
                                        axis=[1,2]
                                    )
                                )
        tf.summary.scalar(
            'reg_loss', tf.reduce_mean(
                self.reg_loss), collections=['train'])

        ips_train_output_j = tf.expand_dims(ips_train_output, axis=1)
        ips_train_output_i = tf.expand_dims(ips_train_output, axis=-1)
        ips_pairwise_logits = ips_train_output_i - ips_train_output_j
        tf.summary.histogram("ips_pairwise_logits", 
                ips_pairwise_logits*pairwise_labels, 
                collections=['train'])

        logits = self.position_bias * self.prob_pos + \
            delta_labels * self.prob_label + self.prob_bias
        self_logits = logits
        if self.hparams.mode == 'adv':
            print('adversarial mode')
            logits += pairwise_logits * self.prob_logits
            self_logits += tf.stop_gradient(ips_pairwise_logits) \
                * self.prob_logits
        elif self.hparams.mode == 'naive':
            print('naive mode')
            logits = 100.0
            self_logits = 100.0
        else:
            print('adversarial mode 2')
            logits += pairwise_logits
            self_logits += tf.stop_gradient(ips_pairwise_logits)

        ips_weights = 1 / (tf.sigmoid(logits) + self.tau)

        if self.hparams.relative_corr:
            print('use relative ips weights')
            ips_weights *= tf.sigmoid(self_logits)

        if self.hparams.self_norm_ips == 1:
            print('self-normalize ips weights via individual sample')
            ips_weights_sum = tf.reduce_sum(pairwise_labels * ips_weights, \
                axis=[1,2],keepdims=True)
            valid_item_count = tf.reduce_sum(pairwise_labels, \
                axis=[1,2],keepdims=True)
            ips_weights_mean = ips_weights_sum / (valid_item_count+self.tau)
            ips_weights = ips_weights / (ips_weights_mean+self.tau)
        elif self.hparams.self_norm_ips == 2:
            print('self-normalize ips weights across mini-batch')
            ips_weights_sum = tf.reduce_sum(pairwise_labels * ips_weights)
            valid_item_count = tf.reduce_sum(pairwise_labels)
            ips_weights_mean = ips_weights_sum / (valid_item_count+self.tau)
            ips_weights = ips_weights / (ips_weights_mean+self.tau)

        tf.summary.scalar(
            'ips_weights_mean', 
            tf.reduce_sum(ips_weights * pairwise_labels) \
                / (tf.reduce_sum(pairwise_labels) + self.tau),
            collections=['train'])

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
        self.loss = self.ips_loss - self.hparams.alpha * self.reg_loss

    def build_variables(self):
        self.position_bias = tf.Variable(
             tf.random.truncated_normal([1, self.rank_list_size, self.rank_list_size],\
                 mean=0.0, stddev=0.1), \
             name = 'position_bias'
            )
        tf.summary.histogram("position_bias", self.position_bias, 
                collections=['train'])         
        self.prob_bias = tf.Variable(0.0, name = 'prob_bias')
        tf.summary.scalar(
            'prob_bias',
            self.prob_bias,
            collections=['train'])
        self.prob_pos = tf.Variable(0.3, name = 'prob_pos')
        tf.summary.scalar(
            'prob_pos',
            self.prob_pos,
            collections=['train'])
        self.prob_logits = tf.Variable(0.6, name = 'prob_logits')
        tf.summary.scalar(
            'prob_logits',
            self.prob_logits,
            collections=['train'])
        self.prob_label = tf.Variable(0.1, name = 'prob_label')
        tf.summary.scalar(
            'prob_label',
            self.prob_label,
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
