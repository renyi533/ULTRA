"""Training and testing the Multileave Gradient Descent (MGD) algorithm for unbiased learning to rank.

See the following paper for more information on the Multileave Gradient Descent (MGD) algorithm.

    * Anne Schuth, Harrie Oosterhuis, Shimon Whiteson, Maarten de Rijke. 2016. Multileave Gradient Descent for Fast Online Learning to Rank. In WSDM. 457-466.

"""

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
from tensorflow.python.framework import ops

import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
from ultra.learning_algorithm.dbgd import DBGD
import ultra.utils
import ultra


class MGD(DBGD):
    """The Multileave Gradient Descent (MGD) algorithm for unbiased learning to rank.

    This class implements the Multileave Gradient Descent (MGD) algorithm based on the input layer feed. See the following paper for more information on the algorithm.

    * Anne Schuth, Harrie Oosterhuis, Shimon Whiteson, Maarten de Rijke. 2016. Multileave Gradient Descent for Fast Online Learning to Rank. In WSDM. 457-466.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Multileave Gradient Descent (DBGD) algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            # The update rate for randomly sampled weights.
            learning_rate=0.05,         # Learning rate.
            max_gradient_norm=5.0,      # Clip gradients to this norm.
            need_interleave=True,       # Set True to use result interleaving
            grad_strategy='sgd',        # Select gradient strategy
            # Select number of rankers to try in each batch.
            ranker_num=4,
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False)
        self.ranker_num = self.hparams.ranker_num

        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        self.winners = tf.placeholder(tf.float32, shape=[None, self.ranker_num + 1],
                                      name="winners")  # winners of interleaved tests
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                    name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                              name="label{0}".format(i)))

        self.global_step = tf.Variable(0, trainable=False)
        self.output = tf.concat(
            self.get_ranking_scores(
                self.docid_inputs,
                is_training=self.is_training,
                scope='ranking_model'),
            1)
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

        # Build model
        if not forward_only:
            self.rank_list_size = exp_settings['train_list_cutoff']
            train_output = tf.concat(
                self.get_ranking_scores(
                    self.docid_inputs,
                    is_training=self.is_training,
                    scope='ranking_model'),
                1)
            train_labels = self.labels[:self.rank_list_size]

            ranking_model_params = self.model.model_parameters
            # new_output_lists = [self.output]
            new_output_lists = []
            params = []
            param_gradient_from_rankers = {}
            # noise_lists = [tf.zeros_like(self.output, tf.float32)]
            for i in range(self.ranker_num):
                # Create random unit noise
                noisy_params = {
                    x: tf.math.l2_normalize(
                        tf.random.normal(
                            ranking_model_params[x].get_shape())) for x in ranking_model_params}
                # Apply the noise to get new ranking scores
                new_output_list = self.get_ranking_scores(
                    self.docid_inputs[:self.rank_list_size], is_training=self.is_training, scope='ranking_model', noisy_params=noisy_params, noise_rate=self.hparams.learning_rate)
                new_output_lists.append(tf.concat(new_output_list, 1))
                for x in noisy_params:
                    if x not in param_gradient_from_rankers:
                        params.append(ranking_model_params[x])
                        param_gradient_from_rankers[x] = [
                            tf.zeros_like(ranking_model_params[x])]
                    param_gradient_from_rankers[x].append(noisy_params[x])

            # Compute NDCG for the old ranking scores.
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            previous_ndcg = ultra.utils.make_ranking_metric_fn('ndcg', self.rank_list_size)(
                reshaped_train_labels, train_output[:, :self.rank_list_size], None)
            self.loss = 1.0 - previous_ndcg

            final_winners = None
            if self.hparams.need_interleave:  # Use result interleaving
                self.output = [self.output, train_output] + new_output_lists
                final_winners = self.winners
            else:  # No result interleaving
                score_lists = [train_output] + new_output_lists
                ndcg_lists = []
                for scores in score_lists:
                    ndcg = ultra.utils.make_ranking_metric_fn(
                        'ndcg', self.rank_list_size)(
                        reshaped_train_labels, scores, None)
                    ndcg_lists.append(ndcg - previous_ndcg)
                ndcg_gains = tf.ceil(tf.stack(ndcg_lists))
                final_winners = ndcg_gains / \
                    (tf.reduce_sum(ndcg_gains, axis=0) + 0.000000001)

            # Compute gradients
            self.gradients = []
            for p in params:
                gradient_matrix = tf.expand_dims(
                    tf.stack(param_gradient_from_rankers[p.name]), axis=0)
                expended_winners = final_winners
                for i in range(gradient_matrix.get_shape().rank -
                               expended_winners.get_shape().rank):
                    expended_winners = tf.expand_dims(
                        expended_winners, axis=-1)
                self.gradients.append(
                    tf.reduce_mean(
                        tf.reduce_sum(
                            expended_winners * gradient_matrix,
                            axis=1
                        ),
                        axis=0)
                )

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            # Gradients and SGD update operation for training the model.
            opt = self.optimizer_func(self.hparams.learning_rate)
            if self.hparams.max_gradient_norm > 0:
                self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                           self.hparams.max_gradient_norm)
                self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                                   global_step=self.global_step)
                tf.summary.scalar(
                    'Gradient Norm',
                    self.norm,
                    collections=['train'])
            else:
                self.norm = None
                self.updates = opt.apply_gradients(zip(self.gradients, params),
                                                   global_step=self.global_step)

            tf.summary.scalar(
                'Learning Rate',
                self.learning_rate,
                collections=['train'])
            tf.summary.scalar('Loss', self.loss, collections=['train'])
            pad_removed_train_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, train_output[:, :self.rank_list_size])
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
        #print ("!!!!!!!!!!!!!", tf.shape(self.new_output))

        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.train_summary  # Summarize statistics.
            ]
            outputs = session.run(output_feed, input_feed)
            # loss, no outputs, summary.
            return outputs[1], None, outputs[-1]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output   # Model outputs
            ]
            outputs = session.run(output_feed, input_feed)
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
