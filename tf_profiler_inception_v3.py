import numpy as np
import os
import cv2
import glob

import tensorflow as tf

# need to in folder of Slim Model from https://github.com/tensorflow/models/blob/master/research/slim
from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing
from nets import nets_factory

from tensorflow.contrib import slim

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

#https://stackoverflow.com/questions/48727264/how-to-convert-numpy-array-image-to-tensorflow-image
def image2placeholder(fundus_image_path, image_size):
    img = cv2.imread(fundus_image_path)
    #print(type(img))
    img = np.array(img)[:, :, 0:3]
    img = cv2.resize(img,dsize=(image_size,image_size), interpolation = cv2.INTER_CUBIC)
    #img = cv2.normalize(img.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    img = img.astype('float')
    img *= 1.0/255.0
    img -= 0.5
    img *= 2.0
    img = np.expand_dims(img,axis=0)
    return img

image_size = inception.inception_v3.default_image_size
is_training = False
#http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
checkpoints = 'inception_v3.ckpt'

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[1,image_size,image_size,3]) 
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(x, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
              
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoints,
        slim.get_model_variables('InceptionV3'))

    results = {}
    with tf.Session() as sess:
        init_fn(sess)

        #profiler
        inception_profiler = model_analyzer.Profiler(graph=sess.graph)        
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        profile_scope_opt_builder = option_builder.ProfileOptionBuilder(
          option_builder.ProfileOptionBuilder.float_operation())
        inception_profiler.profile_name_scope(profile_scope_opt_builder.build())

        #https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg
        for f in (glob.glob("First_Student_IC_school_bus_202076.jpg")):             
            img = image2placeholder(f, image_size)
            probabilities = sess.run(probabilities, feed_dict={x: img},
                                      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                      run_metadata=run_metadata)
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
            for i in range(5):
              index = sorted_inds[i]
              print('Probability %0.2f%% => [%d]' % (probabilities[index] * 100, index))

