"""The basic class that contains all the API needed for the implementation of an unbiased learning to rank algorithm.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tensorflow as tf
from abc import ABC, abstractmethod

import ultra
import ultra.utils as utils


class BaseAlgorithm(ABC):
    """The basic class that contains all the API needed for the
        implementation of an unbiased learning to rank algorithm.

    """
    PADDING_SCORE = -100000

    @abstractmethod
    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        self.is_training = None
        self.docid_inputs = None  # a list of top documents
        self.letor_features = None  # the letor features for the documents
        self.labels = None  # the labels for the documents (e.g., clicks)
        self.output = None  # the ranking scores of the inputs
        # the number of documents considered in each rank list.
        self.rank_list_size = None
        # the maximum number of candidates for each query.
        self.max_candidate_num = None
        self.optimizer_func = tf.train.AdagradOptimizer
        pass

    @abstractmethod
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
        pass

    def remove_padding_for_metric_eval(self, input_id_list, model_output):
        output_scores = tf.unstack(model_output, axis=1)
        if len(output_scores) > len(input_id_list):
            raise AssertionError(
                'Input id list (%d) is shorter than output score list (%d) when remove padding.' \
                   % (len(input_id_list), len(output_scores)))
        # Build mask
        valid_flags = tf.cast(
            tf.concat(
                values=[tf.ones([tf.shape(self.letor_features)[0]]), tf.zeros([1])], axis=0),
            tf.bool
        )
        input_flag_list = []
        for i in range(len(output_scores)):
            input_flag_list.append(
                tf.nn.embedding_lookup(
                    valid_flags, input_id_list[i]))
        # Mask padding documents
        for i in range(len(output_scores)):
            output_scores[i] = tf.where(
                input_flag_list[i],
                output_scores[i],
                tf.ones_like(output_scores[i]) * self.PADDING_SCORE
            )
        return tf.stack(output_scores, axis=1)

    def ranking_model(self, list_size, scope=None, model=None):
        """Construct ranking model with the given list size.

        Args:
            list_size: (int) The top number of documents to consider in the input docids.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.

        """
        output_scores = self.get_ranking_scores(
            self.docid_inputs[:list_size], self.is_training, scope, model)
        return tf.concat(output_scores, 1)

    def get_ranking_scores(self, input_id_list,
                           is_training=False, scope=None, model=None, **kwargs):
        """Compute ranking scores with the given inputs.

        Args:
            input_id_list: (list<tf.Tensor>) A list of tensors containing document ids.
                            Each tensor must have a shape of [None].
            is_training: (bool) A flag indicating whether the model is running in training mode.
            scope: (string) The name of the variable scope.

        Returns:
            A tensor with the same shape of input_docids.

        """
        with tf.variable_scope(scope or "ranking_model"):
            # Build feature padding
            PAD_embed = tf.zeros([1, self.feature_size], dtype=tf.float32)
            letor_features = tf.concat(
                axis=0, values=[
                    self.letor_features, PAD_embed])
            input_feature_list = []
            if not hasattr(self, "model") or self.model is None:
                self.model = utils.find_class(
                    self.exp_settings['ranking_model'])(
                    self.exp_settings['ranking_model_hparams'])
            for i in range(len(input_id_list)):
                input_feature_list.append(
                    tf.nn.embedding_lookup(
                        letor_features, input_id_list[i]))
            if model is None:
                return self.model.build(
                    input_feature_list, is_training=is_training, **kwargs)
            else:
                return model.build(
                    input_feature_list, is_training=is_training, **kwargs)

    def pairwise_cross_entropy_loss(
            self, pos_scores, neg_scores, propensity_weights=None, name=None):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (tf.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (tf.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = tf.ones_like(pos_scores)

        loss = None
        with tf.name_scope(name, "pairwise_cross_entropy_loss", [pos_scores, neg_scores]):
            label_dis = tf.concat(
                [tf.ones_like(pos_scores), tf.zeros_like(neg_scores)], axis=1)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.concat([pos_scores, neg_scores], axis=1), labels=label_dis
            ) * propensity_weights
        return loss

    def sigmoid_loss_on_list(self, output, labels,
                             propensity_weights=None, name=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = tf.ones_like(labels)

        loss = None
        with tf.name_scope(name, "sigmoid_loss", [output]):
            label_dis = tf.math.minimum(labels, 1)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label_dis, logits=output) * propensity_weights
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    def pairwise_loss_on_list(self, output, labels,
                              propensity_weights=None, name=None):
        """Computes pairwise entropy loss.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = tf.ones_like(labels)

        loss = 0.0
        with tf.name_scope(name, "pairwise_loss", [output]):
            sliced_output = tf.unstack(output, axis=1)
            sliced_label = tf.unstack(labels, axis=1)
            sliced_propensity = tf.unstack(propensity_weights, axis=1)
            for i in range(len(sliced_output)):
                cur_propensity = sliced_propensity[i] 
                for j in range(0, len(sliced_output)):
                    cur_label_weight = tf.nn.relu(
                        tf.math.sign(sliced_label[i] - sliced_label[j])
                    )
                    cur_pair_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=cur_label_weight, 
                                        logits=sliced_output[i]-sliced_output[j]
                                    )
                    loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = tf.shape(labels)[0]
        #batch_size = tf.shape(labels[0])[0]
        # / (tf.reduce_sum(propensity_weights)+1)
        return tf.reduce_sum(loss) / tf.cast(batch_size, tf.float32)

    def pair_loss_on_list(self, output, labels,
                          propensity_weights=None, loss_func='hinge', 
                          pair_corr=False, name=None):
        """Computes pairwise entropy loss.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            loss: 'hinge' or 'pair_cross_entropy'
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = tf.ones_like(labels)

        output_i = tf.expand_dims(output, axis=-1)
        output_j = tf.expand_dims(output, axis=1)
        labels_i = tf.expand_dims(labels, axis=-1)
        labels_j = tf.expand_dims(labels, axis=1)
        propensity_i = tf.expand_dims(propensity_weights, axis=-1)
        if not pair_corr:
            print('point ips corr')
            pair_weight = 1.0 - tf.eye(num_rows = tf.shape(labels)[1], 
                                    num_columns=tf.shape(labels)[1], 
                                    batch_shape=[1], 
                                    dtype=tf.dtypes.float32, name=None)
            pair_label = tf.nn.relu(tf.math.sign(labels_i))   
            pair_label -= tf.zeros_like(labels_j)  
        else:
            print('pair ips corr')
            pair_label = tf.nn.relu(tf.math.sign(labels_i-labels_j))
            pair_weight = 1.0

        if loss_func == 'hinge':
            print('hinge loss')
            cur_pair_loss = tf.nn.relu(1.0 + output_j - output_i)
        else:
            print('cross entropy loss')
            cur_pair_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=pair_label,
                                logits=output_i-output_j
                            )                
        loss = cur_pair_loss * pair_weight * propensity_i * pair_label
        return tf.reduce_mean( tf.reduce_sum(loss, axis=[1,2]) )

    def softmax_loss(self, output, labels, propensity_weights=None, name=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (tf.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (tf.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
            name: A string used as the name for this variable scope.

        Returns:
            (tf.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = tf.ones_like(labels)
        loss = None
        with tf.name_scope(name, "softmax_loss", [output]):
            weighted_labels = (labels + 0.0000001) * propensity_weights
            label_dis = weighted_labels / \
                tf.reduce_sum(weighted_labels, 1, keep_dims=True)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=output, labels=label_dis) * tf.reduce_sum(weighted_labels, 1)
        return tf.reduce_sum(loss) / tf.reduce_sum(weighted_labels)

    def normalize_weights(self, propensity, max_propensity_weight=-1):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = tf.unstack(
            propensity, axis=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = tf.stack(pw_list, axis=1)
        if max_propensity_weight > 0:
            propensity_weights = tf.clip_by_value(
                propensity_weights,
                clip_value_min=0,
                clip_value_max=max_propensity_weight)
        return propensity_weights