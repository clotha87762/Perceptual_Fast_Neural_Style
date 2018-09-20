# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import losses
from module import *

class transform(object):
    
    def __init__(self , sess, args ):
        
        self.sess = sess
        
        self.ur = args.ur
        self.gfdim = args.gfdim 
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        
        self.ps = args.pad_size
        
        #self.lr = args.lr
        #self.beta = args.beta
        
        self.deconv = args.deconv
        self.img_size = args.size
        #self.build()
        #self.norm_method = instance_norm # For generation problem use instance norm
        self.batch_size = args.batch
        
        self.is_training = True if args.phase == 'train' else False
        #self.input = tf.placeholder(dtype = float32 , shape = [ self.batch_size , self.img_size , self.img_size , self.in_dim] )
        
        
        if self.ur == 'u1':
            self.generator = self.generator_unet1
        elif self.ur == 'u2':
            self.generator = self.generator_unet2
        else :
            self.generator = self.generator_resnet
        
        #self.layer_dict = layer_dict
        #self.style_dict = style_dict
        #self.content_dict = content_dict 
        
        
        # For ganeration tasks, we use instance norm
    '''
    def build(self , image ):
        
        self.output =  self.generator(image , reuse = False)
        
        if not self.layer_dict == None and not self.content_dict==None and not self.style_dict == None: 
            self.content_loss = losses.content_loss(self.layer_dict , self.content_dict )
            self.style_loss = losses.style_loss(self.layer_dict, self.style_dict)
            self.
        
    def train(self):
        assert 
    
    def inference(self , _input):
        
        
      
        return output
    '''
    
    def generator_unet1(self , _input , reuse = False ): # origin unet
        with tf.variable_scope('generator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            if self.gfdim == None:
                self.gfdim = 8
            
            def conv_block(_input , dim , i):
                x = slim.conv2d(_input , dim , [3,3] , 1 , scope = 'g_e'+str(i)+'_c1')
                x = instance_norm(x , is_training = self.is_training , name = 'g_e'+str(i)+'_b1' )
                x = slim.conv2d( lrelu(x) , dim ,[3,3] , 1 , scope = 'g_e'+str(i)+'_c2')
                x = instance_norm(x , is_training = self.is_training , name = 'g_e'+str(i)+'_b2' )
                x = slim.max_pool2d( lrelu(x) ,[2,2],scope = 'g_e'+str(i)+'_max')
                return x
            
            def up_block(_input, dim , skip ,i ):
                x = slim.conv2d_transpose(_input , dim , [2,2] , scope = 'g_d'+str(i)+'up')
                x = instance_norm(x , is_training = self.is_training , name = 'g_e'+str(i)+'_b1' )
                x = tf.concat( [ relu(x), skip] , axis = -1)
                x = slim.conv2d(x, dim, [1,1], scope = 'g_d'+str(i)+'c1')
                x = instance_norm(x , is_training = self.is_training , name = 'g_e'+str(i)+'_b1' )
                x = slim.conv2d(relu(x), dim, [1,1], scope = 'g_d'+str(i)+'c2')
                x = instance_norm(x , is_training = self.is_training , name = 'g_e'+str(i)+'_b1' )
                return x
            
            _input = tf.pad(_input , [[0,0],[self.ps,self.ps],[self.ps,self.ps],[0,0]], mode = 'REFLECT')
            
            with slim.arg_scope([instance_norm], momentum = 0.9, epsilon = 1e-5):
                d0 = conv_block(_input , self.gfdim , 0)
                d1 = conv_block( d0 , self.gfdim*2 , 1)
                d2 = conv_block( d1 , self.gfdim*4 , 2)
                d3 = conv_block( d2 , self.gfdim*8 , 3)
                d4 = slim.conv2d(d3 , self.gfdim*16 , [1,1] , scope = 'g_e4_c1')
                d4 = instance_norm(d4 , is_training = self.is_training , name = 'g_e4_b1')
                d4 = slim.conv2d( lrelu(d4) , self.gfdim*16 , [1,1] , scope = 'g_e4_c2')
                d4 = instance_norm(d4 , is_training = self.is_training , name = 'g_e4_b2' )
                x = up_block( lrelu(d4), self.gfdim * 8 , d3 ,0)
                x = up_block( relu(x), self.gfdim*4 , d2 , 1)
                x = up_block( relu(x), self.gfdim *2 , d1 , 2)
                x = up_block( relu(x), self.gfdim , d0 , 3)
                x = slim.conv2d( relu(x) , self.out_dim , [1,1] , scope = 'g_predict')
                
                out = tf.nn.tanh(x)
                height = tf.shape(out)[1]
                width = tf.shape(out)[2]
                out = tf.slice(out, [0, self.ps, self.ps, 0], tf.stack([-1, height - self.ps*2, width - self.ps*2, -1]))
                
                return out
           
    
    def generator_unet2(self , _input , reuse = False): # pix2pix unet
        with tf.variable_scope('generator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            if self.gfdim == None:
                self.gfdim = 8
                
            _input = tf.pad(_input , [[0,0],[self.ps,self.ps],[self.ps,self.ps],[0,0]], mode = 'REFLECT')
            
            with slim.arg_scope([instance_norm], momentum = 0.9, epsilon = 1e-5):
                
                
                
                e0 = slim.conv2d(_input , self.gfdim , [2,2] , scope = 'g_e0_conv')
                e0 = instance_norm(e0 , is_training = self.is_training , scope = 'g_e0_bn')
                e1 = slim.conv2d(lrelu(e0) , self.gfdim , [2,2] , scope = 'g_e1_conv')
                e1 = instance_norm(e1 , is_training = self.is_training , scope = 'g_e1_bn')
                e2 = slim.conv2d(lrelu(e1) , self.gfdim , [2,2] , scope = 'g_e2_conv')
                e2 = instance_norm(e2 , is_training = self.is_training , scope = 'g_e2_bn')
                e3 = slim.conv2d(lrelu(e2) , self.gfdim , [2,2] , scope = 'g_e3_conv')
                e3 = instance_norm(e3 , is_training = self.is_training , scope = 'g_e3_bn')
                e4 = slim.conv2d(lrelu(e3) , self.gfdim , [2,2] , scope = 'g_e4_conv')
                e4 = instance_norm(e4 , is_training = self.is_training , scope = 'g_e4_bn')
                e5 = slim.conv2d(lrelu(e4) , self.gfdim , [2,2] , scope = 'g_e5_conv')
                e5 = instance_norm(e5 , is_training = self.is_training , scope = 'g_e5_bn')
                
                d0 = slim.conv2d_transpose(relu(e5) , self.gfdim , [2,2] , scope= 'g_d0_conv' )
                d0 = instance_norm(d0 , is_training = self.is_training , scope = 'g_d0_bn')
                d0 = tf.concat([d0,e4] , axis = -1)
                d1 = slim.conv2d_transpose(relu(d0) , self.gfdim , [2,2] , scope= 'g_d1_conv' )
                d1 = instance_norm(d1 , is_training = self.is_training , scope = 'g_d1_bn')
                d1 = tf.concat([d1,e3] , axis = -1)
                d2 = slim.conv2d_transpose(relu(d1) , self.gfdim , [2,2] , scope= 'g_d2_conv' )
                d2 = instance_norm(d2 , is_training = self.is_training , scope = 'g_d2_bn')
                d2 = tf.concat([d2,e2] , axis = -1)
                d3 = slim.conv2d_transpose(relu(d2) , self.gfdim , [2,2] , scope= 'g_d3_conv' )
                d3 = instance_norm(d3 , is_training = self.is_training , scope = 'g_d3_bn')
                d3 = tf.concat([d3,e1] , axis = -1)
                d4 = slim.conv2d_transpose(relu(d3) , self.gfdim , [2,2] , scope= 'g_d4_conv' )
                d4 = instance_norm(d4 , is_training = self.is_training , scope = 'g_d4_bn')
                d4 = tf.concat([d4,e0] , axis = -1)
                d5 = slim.conv2d_transpose(relu(d4) , self.gfdim , [2,2] , scope= 'g_d5_conv' )
                
                out = tf.nn.tanh(d5)
                height = tf.shape(out)[1]
                width = tf.shape(out)[2]
                out = tf.slice(out, [0, self.ps, self.ps, 0], tf.stack([-1, height - self.ps*2, width - self.ps*2, -1]))
                
                return out
                
             
                
    
    
    def generator_resnet(self , _input, reuse = False): # resnet
        with tf.variable_scope('generator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else :
                assert tf.get_variable_scope().reuse == False
            if self.gfdim == None:
                self.gfdim = 32
                
            def residual(_input , dim,  kernel =[3,3] ,stride = 1  ,scope = 'res_block'):
                x = slim.conv2d(_input, dim, kernel, stride , scope = scope+'_conv1')
                x = slim.conv2d( relu(x) , dim, kernel, stride , scope = scope+'_conv2')
                res = x + _input
                return res
            
            _input = tf.pad(_input , [[0,0],[self.ps,self.ps],[self.ps,self.ps],[0,0]], mode = 'REFLECT')
            
            with slim.arg_scope([instance_norm], momentum = 0.9, epsilon = 1e-5):
                
                c0 = slim.conv2d(_input , self.gfdim , [9,9] , 1 , scope = 'g_e0_conv')
                c0 = relu( instance_norm(c0,is_train = self.is_training ,name ='g_e0_bn' ) )
                
                
                c1 = slim.conv2d( c0 , self.gfdim*2 , [3,3] , 2 , scope = 'g_e1_conv')
                c1 = relu( instance_norm(c1,is_train = self.is_training ,name ='g_e1_bn' ) )
                c2 = slim.conv2d( c1 , self.gfdim*4 , [3,3] , 2 , scope = 'g_e2_conv')
                c2 = relu( instance_norm(c2,is_train = self.is_training ,name ='g_e2_bn' ) )
                
                r0 = residual(c2, self.gfdim*4, [3,3] , 1 , scope = 'g_r0')
                r1 = residual(r0, self.gfdim*4, [3,3] , 1 , scope = 'g_r1')
                r2 = residual(r1, self.gfdim*4, [3,3] , 1 , scope = 'g_r2')
                r3 = residual(r2, self.gfdim*4, [3,3] , 1 , scope = 'g_r3')
                r4 = residual(r3, self.gfdim*4, [3,3] , 1 , scope = 'g_r4')
                
                if self.deconv :
                    d0 = deconv2d(r4 , self.gfdim*2, kernel=3, stride=(2,2), name = 'g_d0_conv')
                else:
                    d0 = deconv2d_resize(r4 , self.gfdim*2, kernel=3, stride=(2,2), name = 'g_d0_conv')
                    
                d0 = relu( instance_norm(d0,is_train = self.is_training , name = 'g_d0_bn'))
                if self.deconv:
                    d1 = deconv2d(d0 , self.gfdim, kernel=3, stride=(2,2), name = 'g_d1_conv')
                else:
                    d1 = deconv2d_resize(d0 , self.gfdim, kernel=3, stride=(2,2), name = 'g_d1_conv')
                d1 = relu( instance_norm(d1,is_train = self.is_training , name = 'g_d1_bn'))
                
                d2 = slim.conv2d(d1 , self.out_dim , [9,9] , 1 , scope = 'g_d2_conv')
                d2 =  instance_norm(d2,is_train = self.is_training , name = 'g_d2_bn')
                
                out = tf.nn.tanh(d2)
                
                height = tf.shape(out)[1]
                width = tf.shape(out)[2]
                out = tf.slice(out, [0, self.ps, self.ps, 0], [-1, height - self.ps*2, width - self.ps*2, -1] )
                
                return out
    
