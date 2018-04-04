# -*- coding: utf-8 -*-

### imports
import matplotlib
#matplotlib.use("Agg")
import shutil
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True, precision=3, linewidth=999)

#-------------------------------------------------------------------------------
# utils 

def broadcast(x, bshape, from_x=None, from_bshape=None):
    shape = x.get_shape().as_list()
    shape = [ i if i is not None else -1 for i in shape]
        
    lbshape = len(bshape)
    if from_bshape is not None:
        lbshape = lbshape - from_bshape

    ones_bshape = [1 for i in range(lbshape)]
    if from_x is None:
        new_shape =  shape + ones_bshape
    else:
        new_shape = shape[:from_x] + ones_bshape + shape[from_x:]
    return tf.reshape(x, new_shape)


#-------------------------------------------------------------------------------
# set the simulation time 

parser = argparse.ArgumentParser() 
parser.add_argument('-t','--time',
        help="Simulation time (sec)",
        action="store", default=3600)  
args = parser.parse_args()

sim_time = int(args.time)
start_time = time.time()

class TimeOverflow(Exception):
    pass

#-------------------------------------------------------------------------------
# only current needed GPU memory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#-------------------------------------------------------------------------------
# load MNIST dataset

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
#shutil.rmtree("MNIST-data")

#-------------------------------------------------------------------------------
# parameters

epochs = 10
data_num = len(train_data)
batch_num = 1000
pattern_num = 28*28
pattern_side = 28
latent_num = 2

k_num = 64
k_side = int(np.sqrt(k_num))

phi_num = 36
phi_side = int(np.sqrt(phi_num))
phi_sigma = 0.2

initial_learning_rate = 0.2

train_data = train_data.reshape(data_num, pattern_side, pattern_side, 1)
test_data = train_data.copy()
np.random.shuffle(test_data)

# latent vector 
x = np.linspace(0,1,k_side)
X,Y = np.meshgrid(x,x)
latents = np.vstack([X.ravel(), Y.ravel()]).T

# phi centroids 
x = np.linspace(0,1,phi_side)
X,Y = np.meshgrid(x,x)
phi_means = np.vstack([X.ravel(), Y.ravel()]).T

#-------------------------------------------------------------------------------
# plotting

# TODO plotting

#------------------------------------------------------------------------------
# main

def phi(x_latent):
    shape = x_latent.shape
    dist = broadcast(x_latent, phi_means.shape, -1, 1) - phi_means
    g = tf.exp(tf.norm(dist, axis=len(dist.shape)-1)**2/(2*phi_sigma**2)) 
    if len(g.shape) > 1: 
        g = tf.transpose(g, [len(g.shape)-1] + range(len(g.shape)-1))
    return g

def y(x, w):
    return tf.matmul(w, phi(x))


# def expectances(t, x, weights, beta):
#     A = (beta / (2.0 * np.pi))**(pattern_num / 2)
#     dists = 
#     return A * tf.exp(-(beta / 2.0) * tf.norm(y(x, weights) - t))**2)
# 
graph = tf.Graph()
with graph.as_default():

    x_latent = tf.placeholder(tf.float32, (None, 2))
   
    weights = tf.get_variable("weights", dtype=tf.float32,
            shape = (pattern_num, phi_num), 
            initializer=tf.random_normal_initializer(stddev=0.02))
   

    g = phi(x_latent)
     
    s = tf.InteractiveSession()
    s.run(tf.global_variables_initializer())
    
    print s.run(y(x_latent, weights), feed_dict={x_latent:latents}) 

    

   #  weights = tf.get_variable("weights", dtype=tf.float32,
   #          shape = (pattern_num, phi_num), 
   #          initializer=tf.random_normal_initializer(stddev=0.02))
   # 
   #  beta = tf.get_variable("beta", initializer=0.2**-1)
   # 
    # patterns = tf.placeholder(tf.float32, (None, pattern_num))
    # 
    # expectances = (beta / (2.0 * np.pi)**int(pattern_num/2)) * \
    #             tf.exp( - (beta/2.0) * \
    #             tf.pow(tf.norm(tf.matmul(weights, phi) -  patterns), 2.0))
    # 
    # y = tf.matmul(weights, phi) 
    # print y.get_shape().as_list()
    # 
    # # #weights = tf.get_variable("weights", shape=(latent_num, pattern_num), initializer = tf.random_normal_initializer(stddev=0.02))
    # # #phi = tf.constant()       
    # #            
    # # with tf.Session(config=config) as session:
    # #     tf.global_variables_initializer().run()
    # #     losses = []
    # #     
    # #     try:
    # #         for epoch in range(epochs):
    # #             
    # #             np.random.shuffle(train_data)
    # #                         
    # #             # run a batch of train steps 
    # #             elosses = []
    # #             for batch in range(data_num//batch_num):
    # # 
    # #                 curr_time = time.time() - start_time
    # #                 if curr_time >= sim_time:
    # #                     raise TimeOverflow("No time left!")
    # #                 
    # #                 print "epoch:%4d       batch:%4d" % (epoch, batch)
    # # 
    # #                 # TODO: step
    # #                 
    # #                 elosses.append(loss_)
    # # 
    # # 
    # #             losses.append(np.mean(elosses))
    # # 
    # #     except TimeOverflow:
    # # 
    # #         print "Plotting partial results..."
    # #     
    # #             
