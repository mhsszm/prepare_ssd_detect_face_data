""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types, iteritems


import os
import sys
current_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(current_path)
from base_function import *
# import types 
import cv2
# import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
# import math
# logging_path = os.path.join(current_path,"..","lib")
# sys.path.append(logging_path)
# from logger import *
def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item() #pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             inp,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc


    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax
    
class PNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='PReLU1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='PReLU2')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
             .softmax(3,name='prob1'))

        (self.feed('PReLU3') #pylint: disable=no-value-for-parameter
             .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))
        
class RNet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .fc(128, relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(2, relu=False, name='conv5-1')
             .softmax(1,name='prob1'))

        (self.feed('prelu4') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv5-2'))

class ONet(Network):
    def setup(self):
        (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(10, relu=False, name='conv6-3'))



class mtcnn(object):    
    def __init__(self, model_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "models"),
      minsize = 60, maxsize = 0, threshold = [ 0.6, 0.8, 0.95], factor = 0.709):
        # if maxsize < minsize then maxsize is invalid.

        self.minsize = minsize # minmum size of face
        self.maxsize = maxsize # maxmum size of face
        self.threshold = threshold # new threshold
        self.factor = factor # scale factor
        self.__version__ = '2018-08-01'

        if not model_path:
            model_path,_ = os.path.split(os.path.realpath(__file__))
            print (model_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            with tf.variable_scope('pnet'):
                data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
                pnet = PNet({'data':data})
                pnet.load(os.path.join(model_path, 'det1.npy'), sess)
            with tf.variable_scope('rnet'):
                data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
                rnet = RNet({'data':data})
                rnet.load(os.path.join(model_path, 'det2.npy'), sess)
            with tf.variable_scope('onet'):
                data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
                onet = ONet({'data':data})
                onet.load(os.path.join(model_path, 'det3.npy'), sess)
            
        self.pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        self.rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        self.onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})
   
    def detect_face_cropped(self, img_file, intput_type = 0, output_type = 0, margin = 15):
        # t1 = time.time()
        #status      :   0 -- no face
        #               -6 -- have face
        #cropped_faces: [] -- no face
        #               not [] -- little face
        bounding_boxes, points ,status = self.detect_face(img_file, intput_type , output_type,margin)
        if (not (type(bounding_boxes) is np.ndarray)) or bounding_boxes.shape[0]==0:
          # print ("no result return! ","status ",str(status))
          cropped_faces = None
        else:
          cropped_faces = get_cropped_face(img_file, bounding_boxes)
        # logger.info("mtcnn cost time {} ! ".format(time.time() - t1))
        return cropped_faces,status

    def detect_face(self, img_file, intput_type = 0, output_type = 0, margin = 15):    
        # img_file: input image
        # intput_type : 0 -- get image path
        #               1 -- get image data (cv2 read)
        #               2 -- get image data (cv2 read and used cv2.COLOR_BGR2RGB)
        
        # output_type:  0 -- get all result
        #               1 -- get the center one 
        #               2 -- get the largest areas one

        # margin: expand_size

        if intput_type == 0:
          img_file = cv2.imread(img_file)
          img=cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        elif intput_type == 1:
          img=cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
        elif intput_type == 2:
          img = img_file

        img_h=img.shape[0]
        img_w=img.shape[1]
        m_scales = get_scales(img_h, img_w, self.minsize, self.maxsize, self.factor)

        # pnet_t1=time.time()
        pnet_total_boxes = pnet_detect(img.copy(), img_h, img_w, self.pnet_fun, m_scales, self.threshold[0], boxes_method = 'Nms')
        # print ("pnet_time >>>>>>> ",time.time()-pnet_t1)
        if pnet_total_boxes.shape[0] >0:
            pnet_total_boxes =pnet_processing(pnet_total_boxes.copy())
            # return pnet_total_boxes, None
            pnet_temp_img = make_data(img, pnet_total_boxes, img_w, img_h, 24)
            rnet_total_boxes = rnet_detect(pnet_temp_img, pnet_total_boxes, self.rnet_fun, self.threshold[1])
            if rnet_total_boxes.shape[0] >0:
                # return rnet_total_boxes, None
                rnet_temp_img = make_data(img, rnet_total_boxes, img_w, img_h, 48)
                onet_total_boxes, points = onet_detect(rnet_temp_img, rnet_total_boxes, self.onet_fun, self.threshold[2])
                if onet_total_boxes.shape[0] >0:
                  # print (onet_total_boxes)
                  if margin > 0:
                    onet_total_boxes = expand_face(onet_total_boxes, img_h, img_w, margin)
                  if output_type == 0:
                    return onet_total_boxes, points,-6
                  elif output_type == 1:
                    boxes,points = get_center_result(img_h, img_w, onet_total_boxes, points)
                    return boxes,points,-6
                  else:
                    boxes,points = get_max_areas_result(img_h, img_w, onet_total_boxes, points)
                    return boxes,points,-6
                else:
                    return None, None,0
            else:
                return None, None,0
        else:
            return None, None,0