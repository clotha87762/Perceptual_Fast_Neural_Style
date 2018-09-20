# -*- coding: utf-8 -*-


exclude_dict = { 'vgg_16' : ['vgg_16/fc'],
                 'resnet_v2_50' : [],
                 'inception_v3' : []
                }

content_loss_dict = { 'vgg_16' : ["vgg_16/conv3/conv3_3"],
                      'resnet_v2_50' : [],
                      'inception_v3' : []
                     }


style_loss_dict = {
                    'vgg_16' : ["vgg_16/conv1/conv1_2",
                                "vgg_16/conv2/conv2_2",
                                "vgg_16/conv3/conv3_3",
                                "vgg_16/conv4/conv4_3"],
                    'resnet_v2_50' : [],
                    'inception_v3' : []
        }