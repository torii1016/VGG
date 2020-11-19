# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import pickle

from .tf_util import Layers, NetworkCreater

class _vgg16_network(Layers):
    def __init__(self, name_scopes, config):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)

        self._vgg16_config = config["VGG16"]
        self._network_creater = NetworkCreater(self._vgg16_config, name_scopes[0]) 

    def set_model(self, inputs, is_training=True, reuse=False):
        return self._network_creater.create(inputs, self._vgg16_config, is_training, reuse)


class VGG(object):
    
    def __init__(self, param, config, image_info, output_dim):
        self._lr = param["lr"]
        self._image_width, self._image_height, self._image_channels = image_info
        self._output_dim = output_dim

        self._network = _vgg16_network([config["VGG16"]["network"]["name"]], config)


    def set_model(self):
        self._set_network()
        self._set_loss()
        self._set_optimizer()


    def _set_network(self):
        
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self._image_height, self._image_width, self._image_channels])
        self.gt_val = tf.compat.v1.placeholder(tf.float32, [None, self._output_dim])

        self._logits = self._network.set_model(self.input, is_training=True, reuse=False) # train
        self._logits_wo = self._network.set_model(self.input, is_training=False, reuse=True) # inference
        self._output_probability = tf.nn.softmax(self._logits_wo)

        self._correct_prediction = tf.equal(tf.argmax(self._logits_wo,1), tf.argmax(self.gt_val,1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))


    def _set_loss(self):

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, 
                                                       labels=self.gt_val)
        self._loss_op = tf.reduce_mean(loss)


    def _set_optimizer(self):
        #self._train_op = tf.compat.v1.train.RMSPropOptimizer(self._lr).minimize(self._loss_op, var_list=self._network.get_variables())
        self._train_op = tf.compat.v1.train.AdamOptimizer(self._lr).minimize(self._loss_op)


    def train(self, sess, input_images, input_labels):
        feed_dict = {self.input: input_images, self.gt_val: input_labels}
        loss, _, logits = sess.run([self._loss_op, self._train_op, self._logits], feed_dict=feed_dict)
        return loss, logits


    def inference(self, sess, input_image):
        feed_dict = {self.input: [input_image]}
        logits = sess.run([self._output_probability], feed_dict=feed_dict)
        return np.squeeze(logits)  # remove extra dimension


    def test(self, sess, input_images, input_labels):
        feed_dict = {self.input: input_images, self.gt_val: input_labels}
        accuracy = sess.run([self._accuracy], feed_dict=feed_dict)
        return np.squeeze(accuracy)  # remove extra dimension