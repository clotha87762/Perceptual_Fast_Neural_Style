# -*- coding: utf-8 -*-


exclude_dict = { 'vgg_16' : ['vgg_16/fc'],
                 'resnet_v2_50' : [],
                 'InceptionV3' : ["InceptionV3/Logits",
                                  "InceptionV3/AuxLogits"
                                  ]
                }

content_loss_dict = { 'vgg_16' : ["vgg_16/conv3/conv3_3"],
                      'resnet_v2_50' : [],
                      'InceptionV3' : [
                              "Mixed_5c",
                              
                              ]
                     }


style_loss_dict = {
                    'vgg_16' : ["vgg_16/conv1/conv1_2",
                                "vgg_16/conv2/conv2_2",
                                "vgg_16/conv3/conv3_3",
                                "vgg_16/conv4/conv4_3"],
                    'resnet_v2_50' : [],
                    'InceptionV3' : [
                                     "Conv2d_1a_3x3",
                                      "Conv2d_2a_3x3",
                                      "Conv2d_2b_3x3",
                                      'Conv2d_3b_1x1',
                                      'Conv2d_4a_3x3',
                                      "Mixed_5b",
                                      "Mixed_5c"
                                      ]
        }


'''
'vgg_16' : ["vgg_16/conv1/conv1_2",
                                "vgg_16/conv2/conv2_2",
                                "vgg_16/conv3/conv3_3",
                                "vgg_16/conv4/conv4_3"],
'''