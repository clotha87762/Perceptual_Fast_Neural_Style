# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import numpy as np
import os
import preprocessing.preprocessing_factory
import nets.nets_factory
import random
import loss_config
from time import time
from glob import glob
import reader
import losses
from transform import transform
import matplotlib.pyplot as plt
import scipy.misc

parser = argparse.ArgumentParser(description='')

parser.add_argument('--style_dir' , dest = 'style_dir' ,  help = 'path of style image')
parser.add_argument('--save_dir' , dest = 'save_dir' , default = './save' , help = 'path of saved images')
parser.add_argument('--sample_dir' , dest = 'sample_dir' , default = './sample' , help = 'path of saved images')
parser.add_argument('--target_dir' , dest = 'target_dir' , default = './target' , help = 'path of target images')
parser.add_argument('--test_dir', dest = 'test_dir' , default = './test' , help = 'path of test images')

parser.add_argument('--loss_model', dest = 'loss_model' , default = 'vgg_16' , help = 'name of the network, please refer\
                    to nets/nets_factory.py' )

parser.add_argument('--log_dir' , dest = 'log_dir' , default = './logs' , help = 'path of tensorboard logs')
parser.add_argument('--ckpt_dir' , dest = 'ckpt_dir' , default = './checkpoints' , help = 'path of ckpt files')

parser.add_argument('--phase', dest = 'phase' , default = 'train'  , help = 'train or test' )
parser.add_argument('--size', dest = 'size' , type = int  , default = 256,  help = 'image size' )
parser.add_argument('--epoch' , dest= 'epoch' , type = int , default = 10 )
parser.add_argument('--batch' , dest = 'batch' , type = int , default = 4 )
parser.add_argument('--in_dim' , dest = 'in_dim' , type = int , default = 3 )
parser.add_argument('--out_dim' , dest = 'out_dim' , type = int , default = 3 )

parser.add_argument('--ur' , dest = 'ur' ,  default = 'r' , help= 'generator is unet or resnet, u1/u2/r ' )
parser.add_argument('--gfdim' , dest = 'gfdim' , type = int  ,default = None , help= 'first layer dim of generator' )

parser.add_argument('--lr' , dest = 'lr' ,  type = int , default = 0.0002 , help= 'init learning rate of adam' )
parser.add_argument('--beta' , dest = 'beta' ,  type = int , default = 0.5  , help= 'beta1 of adam' )
parser.add_argument('--deconv' , dest = 'deconv' ,  type = bool , default = False  , help= 'Use deconv or resize-conv' )

parser.add_argument('--pad_size' , dest = 'pad_size' , type=int, default = 10 , help = 'pad size before & after feeding into network')

parser.add_argument('--save_freq', dest = 'save_freq' , type=int , default = 10 , help = 'frequency to save model')
parser.add_argument('--c_weight' , dest = 'c_weight' , type = float , default = 1.0 , help = 'weight of content loss' )
parser.add_argument('--s_weight' , dest = 's_weight' , type = float , default = 100.0 , help = 'weight of style loss' )
parser.add_argument('--tv_weight' , dest = 'tv_weight' , type = float , default = 0.001 , help = 'weight of tv loss' )


args = parser.parse_args()

slim = tf.contrib.slim

def main(_):
    
    if not os.path.exists(args.style_dir):
        os.makedirs(args.style_dir)
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists('./pretrained'):
    	os.makedirs('./pretrained')
        
    style_name = (args.style_dir.split('/')[-1]).split('.')[0]
    if not os.path.exists( os.path.join(args.ckpt_dir, style_name ) ):
        os.makedirs(os.path.join(args.ckpt_dir, style_name ))
    
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    
    if args.phase == 'train':
        train()
    else:
        evaluate()
    

def train():
    
    style_feature , style_grams = get_style_feature()
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
        
            #loss_input_style = tf.placeholder(dtype = tf.float32 , shape = [args.batch , args.size , args.size , args.in_dim ])
            #loss_input_target =tf.placeholder(dtype = tf.float32 , shape = [args.batch , args.size , args.size , args.in_dim ])
            
            # For online optimization problem, use testing preprocess for both train and test
            preprocess_func , unprocess_func = preprocessing.preprocessing_factory.get_preprocessing( args.loss_model , is_training = False )
            
            
            
            images = reader.image(args.batch, args.size , args.size, args.target_dir , preprocess_func, \
                                 args.epoch , shuffle = True)
            
            
        
        
            model = transform(sess,args)
            transformed_images = model.generator(images, reuse = False)
            
            #print('qqq')
            #print( tf.shape(transformed_images).eval())
           
            unprocess_transform = [ (img) for img in tf.unstack( transformed_images , axis=0, num=args.batch) ]
            
            processed_generated = [ preprocess_func(img ,args.size , args.size) for img in unprocess_transform]
            processed_generated = tf.stack(processed_generated)
            
            loss_model = nets.nets_factory.get_network_fn(args.loss_model ,num_classes = 1,is_training = False)
            
            
            pair = tf.concat([processed_generated , images] , axis = 0 )
            _ , end_dicts = loss_model( pair , spatial_squeeze = False)
             
            init_loss_model = load_pretrained_weight(args.loss_model)
            
           
            
            c_loss = losses.content_loss(end_dicts , loss_config.content_loss_dict[args.loss_model])
            
            s_loss  , s_loss_sum = losses.style_loss(end_dicts, loss_config.style_loss_dict[args.loss_model] ,style_grams)
            
            tv_loss = losses.total_variation_loss(transformed_images)
            
            loss = args.c_weight * c_loss +  args.s_weight * s_loss +  args.tv_weight * tv_loss
            
           
            print('shapes')
            print(pair.get_shape())
            
            tf.summary.scalar('average', tf.reduce_mean(images))
            tf.summary.scalar('gram average', tf.reduce_mean(tf.stack(style_feature)))
            
            tf.summary.scalar('losses/content_loss', c_loss)
            tf.summary.scalar('losses/style_loss', s_loss)
            tf.summary.scalar('losses/tv_loss', tv_loss)
    
            tf.summary.scalar('weighted_losses/weighted_content_loss', c_loss * args.c_weight)
            tf.summary.scalar('weighted_losses/weighted_style_loss', s_loss * args.s_weight)
            tf.summary.scalar('weighted_losses/weighted_regularizer_loss', tv_loss * args.tv_weight)
            tf.summary.scalar('total_loss', loss)
    
            for layer in loss_config.style_loss_dict[args.loss_model]:
                tf.summary.scalar('style_losses/' + layer, s_loss_sum[layer])
                
            tf.summary.image('transformed',  tf.stack(unprocess_transform,axis=0) )
            # tf.image_summary('processed_generated', processed_generated)  # May be better?
            tf.summary.image('ori', tf.stack([
                    unprocess_func(image) for image in tf.unstack( images, axis=0, num=args.batch)
                ]))
        
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(args.log_dir)
            
            step = tf.Variable( 0 , name = 'global_step' , trainable=False)
            
            
            all_trainables = tf.trainable_variables()
            all_vars =  tf.global_variables()
            to_train = [var for var in all_trainables if not args.loss_model in var.name]
            to_restore = [var for var in all_vars if not args.loss_model in var.name ]
            
                
            optim = tf.train.AdamOptimizer( learning_rate = args.lr , beta1 = args.beta).minimize(\
                                           loss = loss , var_list = to_train , global_step = step)
            
            
            saver = tf.train.Saver(to_restore)
            style_name = (args.style_dir.split('/')[-1]).split('.')[0]
            
            ckpt = tf.train.latest_checkpoint(os.path.join(args.ckpt_dir,style_name))
            if ckpt:
                tf.logging.info('Restoring model from {}'.format(ckpt))
                saver.restore(sess, ckpt)
            
            sess.run( [ tf.global_variables_initializer() , tf.local_variables_initializer()])
            #sess.run(init_loss_model)
            init_loss_model(sess)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            start_time = time()
            #i = 0
            try:
                while True:
                
                    _ , gs, sum_info,c_info, s_info, tv_info, loss_info = sess.run( [optim , step , summary ,c_loss , s_loss , tv_loss , loss] )
                    writer.add_summary(sum_info , gs)
                    elapsed = time() - start_time
                    
                    print(gs)
                    
                    if gs % 10 == 0:
                        tf.logging.info('step: %d, c_loss %f  s_loss %f  tv_loss %f total Loss %f, secs/step: %f' % (gs, c_info, s_info, tv_info, loss_info, elapsed))
                    
                    if gs % args.save_freq == 0:
                        saver.save(sess, os.path.join(args.ckpt_dir,style_name,style_name+'.ckpt'))
                        
            except tf.errors.OutOfRangeError:
                print('run out of images!  save final model: ' + os.path.join(args.ckpt_dir,style_name+'.ckpt-done') )
                saver.save(sess , os.path.join(args.ckpt_dir, style_name ,style_name+'.ckpt-done') )
                tf.logging.info('Done -- file ran out of range')
            finally:
                coord.request_stop()
            
            coord.join(threads)
            
            print('end training')
            '''
            #only support jpg and png
            style_image_name = glob('./{}/*.jpg'.format(args.style_dir)) + glob('./{}/*.png'.format(args.style_dir))
            target_image_name = glob('./{}/*.jpg'.format(args.target_dir)) + glob('./{}/*.png'.format(args.target_dir))
            
            
            for i in range(args.epoch):
                
                random.shuffle(target_image_name)
                
                for j in range(args.batch):
                    
                    batch_name = target_image_name[j*args.batch : (j+1)*args.batch] if not j==args.batch-1 \
                    else target_image_name[j*args.batch :]
            '''
            

def load_pretrained_weight(name):
    
    to_exclude = loss_config.exclude_dict
    #print(name)
    vars_to_restore = slim.get_variables_to_restore( include = [name] , exclude = to_exclude[name] )
    
    #print('---vars to restore---')
    #print(vars_to_restore)
    
    return slim.assign_from_checkpoint_fn(
        './pretrained/{}/{}'.format(name, name+'.ckpt'),
        vars_to_restore,
        ignore_missing_vars=True)


# get the style feature of the style image once for all to speedup 
def get_style_feature():
    
    with tf.Graph().as_default():
       
            
            preprocess_func , unprocess_func = preprocessing.preprocessing_factory.get_preprocessing( args.loss_model , is_training = False )
            
            style_img = reader.get_image(args.style_dir, args.size, args.size, preprocess_func)
            style_img = tf.expand_dims( style_img , 0 )
            
            loss_model = nets.nets_factory.get_network_fn(args.loss_model,1,is_training = False)
            
            _ , end_dict = loss_model(style_img, spatial_squeeze = False)
            
            init_loss_model = load_pretrained_weight(args.loss_model)
            #init_loss_model2 = load_pretrained_weight(args.loss_model)
            
            features = []
            feature_grams = []
            
            #sess.run([tf.global_variables_initializer() , tf.local_variables_initializer()])
            #sess.run([init_loss_model])
            
            
            for layer in loss_config.style_loss_dict[args.loss_model]:
                
                #print('--layer--' + layer)
                feature =  end_dict[layer] 
                gram =  losses.gram_matrix(feature) 
                
                feature_s = tf.squeeze(feature,[0])
                gram_s = tf.squeeze(gram, [0] )
                
                #f , g = sess.run([ feature_s , gram_s ])
                
                features.append( feature_s)
                feature_grams.append( gram_s )
                
                
            with tf.Session() as sess:
                
                init_loss_model(sess)
                ff , gg = sess.run( [features , feature_grams ] )
                #print('qwq')
                return ff , gg
        


def evaluate():
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
        
            #loss_input_style = tf.placeholder(dtype = tf.float32 , shape = [args.batch , args.size , args.size , args.in_dim ])
            #loss_input_target =tf.placeholder(dtype = tf.float32 , shape = [args.batch , args.size , args.size , args.in_dim ])
            
            # For online optimization problem, use testing preprocess for both train and test
            preprocess_func , unprocess_func = preprocessing.preprocessing_factory.get_preprocessing( args.loss_model , is_training = False )
            
            
            images = reader.image( 1 , args.size , args.size, args.test_dir , preprocess_func, \
                                  1 , shuffle = False)
            
                
            model = transform(sess,args)
            transformed_images = model.generator(images, reuse = False)
            unprocess_transform = [ unprocess_func(img) for img in tf.unstack( transformed_images , axis=0, num=args.batch) ]
            
            all_vars =  tf.global_variables()
            to_restore = [var for var in all_vars if not args.loss_model in var.name ]
            
            sess.run([tf.global_variables_initializer() , tf.local_variables_initializer()])

            saver = tf.train.Saver(to_restore)
            style_name = (args.style_dir.split('/')[-1]).split('.')[0]
            
            ckpt = tf.train.latest_checkpoint(os.path.join(args.ckpt_dir,style_name))
            if ckpt:
                tf.logging.info('Restoring model from {}'.format(ckpt))
                saver.restore(sess, ckpt)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            start_time = time()
            i = 0
            try:
                while True:
                
                    images = sess.run( unprocess_transform )
                    for img in images:
                        path = os.path.join(args.save_dir, str(i)+'.jpg' )
                        scipy.misc.imsave( path, img )
                        i = i+1
                    
            except tf.errors.OutOfRangeError:
                print('eval finished')
            finally:
                coord.request_stop()
            
            coord.join(threads)
            
            
    
    

if __name__ == '__main__':
    tf.app.run()
