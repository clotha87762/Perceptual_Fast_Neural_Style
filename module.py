# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

def log2(_input):
    numerator = tf.log(_input)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def dropout(_input , prob = 0.5 ):
    return tf.nn.dropout(_input,prob)

def relu(x):
    return tf.nn.relu(x)

def lrelu(x):
    return tf.nn.leaky_relu(x)

#def lrelu(_input, leak = 0.2):
#    lrelu =  tf.maximum(_input, _input*leak)
#    #nan_to_zero = tf.where(tf.equal(lrelu, lrelu), lrelu, tf.zeros_like(lrelu))
#    return lrelu#nan_to_zero


#def relu(_input):
#    relu = tf.nn.relu(_input)
    
    #nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
#    return relu#nan_to_zero


def instance_norm( _input, name="instance_norm" , is_train = True,epsilon=1e-5, momentum = 0.9):
    with tf.variable_scope(name):
        depth = _input.get_shape()[-1]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(_input, axes=[1,2], keep_dims=True)
        epsilon = epsilon
        inv = tf.rsqrt(variance + epsilon)
        normalized = (_input-mean)*inv
        return scale*normalized + offset

    
def batch_norm(_input, is_train = True,epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(_input, decay= momentum,                                                                                                               
              is_training=is_train, scale=True, updates_collections=None, scope= name, epsilon = epsilon,
              reuse= False)
        
def res_block1 (_input , num_filters, kernel = 3, stride =(2,2) , pad = 'SAME', init_std=0.05 , name ='res1'):
    with tf.variable_scope(name):
        conv1 = conv2d(_input, num_filters,  kernel=kernel, stride = stride , pad = pad , init_std = init_std , name = name+'_c1')
        conv2 = conv2d(relu(conv1), num_filters,  kernel=kernel, stride = stride , pad = pad , init_std = init_std , name = name+'_c2')

        residual = _input + conv2

        return residual
    

def res_block2 (_input , num_filters, kernel = 3, stride =(2,2) , pad = 'SAME', init_std=0.05 , name ='res2'):
    with tf.variable_scope(name):
        p = int((kernel - 1) / 2)
        y = tf.pad(_input, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_norm(conv2d(y, num_filters, kernel, stride, pad="VALID", name=name+'_c1'), name+'_bn1')
        y = tf.pad(relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_norm(conv2d(y, num_filters, kernel, stride, pad = "VALID" , name=name+'_c2'), name+'_bn2')
        return y + _input # identity addition
    


    


def conv2d( _input , num_filter , kernel = 5 , stride=(2,2) , pad = 'SAME' , init_std = 0.05,
           name = 'conv2d'):
    
    with tf.variable_scope(name):
        weight = tf.get_variable(name ='weight' , shape=[kernel,kernel, _input.shape[-1] , num_filter] , initializer=tf.truncated_normal_initializer(stddev=init_std , dtype= tf.float32))
        conv = tf.nn.conv2d( _input, weight, strides=[1, stride[0], stride[1],1] , padding=pad)
        bias = tf.get_variable(name='bias',shape=[num_filter], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv,bias)
        output = tf.reshape(output, tf.shape(conv))
        return output


def deconv2d( _input , output_shape, kernel = 5 , stride=(2,2) , pad = 'SAME' , init_std = 0.05,
           name = 'deconv2d'):
     with tf.variable_scope(name):
        weight = tf.get_variable(name ='weight' , shape=[kernel,kernel, output_shape[-1] , _input.get_shape()[-1]] , initializer=tf.random_normal_initializer(stddev=init_std,dtype=tf.float32))
        
        deconv = tf.nn.conv2d_transpose(_input , weight, output_shape=output_shape, strides=[1,stride[0],stride[1],1] , padding = pad)
        bias = tf.get_variable(name='bias',shape=[output_shape[-1]], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(deconv,bias)
        output = tf.reshape(output, tf.shape(deconv))
        
        return output

def deconv2d_resize(_input ,num_filter , kernel = 5 , stride=(2,2) , pad = 'SAME' , init_std = 0.05, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
           name = 'deconv2d_resize'):  # use resize before a normal convolution to upsample, avoiding checkerboard effect
    with tf.variable_scope(name):
        weight = tf.get_variable(name ='weight', shape=[kernel,kernel,_input.get_shape()[-1] , num_filter], initializer=tf.truncated_normal_initializer(stddev=init_std , dtype= tf.float32))
        
        width = tf.shape(_input)[2]
        height = tf.shape(_input)[1]
        
        r_width = width*stride[1]*2
        r_height = height*stride[0]*2
        
        resize_img = tf.image.resize_images(_input, [r_height,r_width] , method = method)
        conv = tf.nn.conv2d(resize_img, weight, strides=[ 1, stride[0], stride[1], 1], padding = pad)
        bias = tf.get_variable(name='bias', shape=[num_filter], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, bias)
        output = tf.reshape(output, tf.shape(conv))
        
        return output


def dense(_input, dim, init_std=0.02 , init_bias = 0.0 , name  = 'dense' ):
    
    #t = tf.shape(_input)[1]
    #u = tf.stack( [t,dim]  )
    #print('----')
    #print(u)
    shape = _input.get_shape().as_list()

    with tf.variable_scope(name):
        
        #output = tf.layers.dense(_input , dim , kernel_initializer = tf.truncated_normal_initializer(stddev=init_std) , trainable = True)
        
        weight = tf.get_variable(name='weight', shape= [shape[1] , dim] , dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=init_std))
        bias = tf.get_variable(name='bias', shape=[dim], initializer=tf.constant_initializer(init_bias))
        output = tf.matmul(_input, weight)
        output = output + bias
        return output
        
    
    
