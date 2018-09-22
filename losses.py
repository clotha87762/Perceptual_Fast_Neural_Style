# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def gram_matrix( x ):
    shape = tf.shape(x)
    filters = tf.reshape(x , [shape[0] , -1 , shape[3]])
    gram = tf.matmul( filters , filters , transpose_a = True )
    gram = gram / tf.to_float( shape[1] * shape[2] * shape[3])
    return gram


def content_loss(layer_dict , content_layer ):
    
    total = 0
    
    for layer in content_layer:
        out = layer_dict[layer]
        generated , origin = tf.split(out, 2 , axis = 0)
        t = generated - origin
        total += tf.reduce_mean( t**2 )
        
    
    return total


def style_loss(layer_dict , style_layer , style_features):
    
    total = 0
    loss_sum = {}
    
    for layer  , style_feature in zip(style_layer , style_features):
        
        print('----')
        print(layer)
        
        out = layer_dict[layer]
        
        generated , _ = tf.split(out, 2 , axis = 0)
        
        image_count = tf.shape(generated)[0]
        
        gen_gram = gram_matrix( generated )
        #ori_gram = gram_matrix( origin )
        #print(gen_gram.get_shape())
        
        s = tf.expand_dims(style_feature,0)
        style_gram = tf.tile(s, [image_count,1,1])
        #print(style_gram.get_shape())
        
        t = gen_gram - style_gram
        #t = gen_gram - s
        loss = tf.reduce_mean( t**2 )
        total += loss
        loss_sum[layer] = loss
        
    return total , loss_sum


def total_variation_loss ( image ):
    
    shape = tf.shape(image)
    height = shape[1] - 1
    width = shape[2] - 1
    # compute x and y variation together
    #v1 = tf.slice( image , [0 , 1 , 1 ,0] , [-1 , height , width , -1])
    #v2 = tf.slice( image , [0, 0, 0, 0] , [-1, height, width, -1] )
    #tv = v1 - v2
    #loss = tf.reduce_mean(tv**2)
    
    # compute x and y variation separately
    v1 = tf.slice( image , [0, 1, 0, 0] , [-1,-1 ,-1 ,-1]) - tf.slice(image , [0,0,0,0] ,[-1,height,-1,-1])
    v2 = tf.slice( image , [0, 0, 1, 0] , [-1,-1 ,-1 ,-1]) - tf.slice(image , [0,0,0,0] ,[-1,-1,width,-1])
    loss = tf.reduce_mean(v1**2) + tf.reduce_mean(v2**2)
    return loss




