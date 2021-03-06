# -*- coding: utf-8 -*-

### imports
import matplotlib
matplotlib.use("Agg")
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
    """
    Expands to arbitrary dimensions so to broadcast 
    with another tensor

    :param x: tensor to expand for broadcasting
    :param bshape: shape of the tensor to broadcast with 
    :from_x: index in the shape of x from which insert the vector of 
            singular dimensions
    :from_bshape: 
    """
    # convert shape for reshaping, substituting 
    # unknown dimensions with -1
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
    res = tf.reshape(x, new_shape)
    return res

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
train_data = mnist.train.images[:10000] # Returns np.array


#shutil.rmtree("MNIST-data")

#-------------------------------------------------------------------------------
# parameters

epochs = 1000
data_size = len(train_data)
batch_size = data_size/10
pattern_size = 28*28
pattern_side = 28
latent_size = 2

train_data = train_data - train_data.min(1).reshape(data_size, 1)
train_data = train_data/train_data.sum(1).reshape(data_size, 1)

k_size =20*20
k_side = int(np.sqrt(k_size))
phi_sigma = 0.02

learning_rate = 0.002
init_w_scale = 0.0002

test_data = train_data.copy()
np.random.shuffle(test_data)

# latent vector 
x = np.linspace(0,1,k_side)
X,Y = np.meshgrid(x,x)
latents = np.vstack([X.ravel(), Y.ravel()]).T

# phi centroids 
x = np.linspace(0,1,k_side)
X,Y = np.meshgrid(x,x)
phi_means = np.vstack([X.ravel(), Y.ravel()]).T

#-------------------------------------------------------------------------------
# plotting

# TODO plotting

def plot_weights(W, session):
    # plot weights

    W_ = W.eval()  
    minw = np.min(W_)
    maxw = np.max(W_)
    W_ = (W_- minw)/(maxw-minw) 
    fig = plt.figure(figsize=(10, 10))
    kk = int(np.sqrt(W_.shape[1]))
    side = int(np.sqrt(W_.shape[0]))

    for i in range(kk):
        for j in range(kk):
            p = i*kk + j
            ax = fig.add_subplot(kk,kk, p+1)
            ax.imshow(W_[:,p].reshape(side, side))
            ax.set_axis_off()
    fig.canvas.draw()
    fig.savefig("weights.png")
    plt.close(fig)

def plot_patterns(patterns, session):

    fig = plt.figure(figsize=(10, 10))
    kk = int(np.sqrt(patterns.shape[0]))
    side = int(np.sqrt(patterns.shape[1]))
    

    for i in range(kk):
        for j in range(kk):
            p = i*kk + j
            ax = fig.add_subplot(kk,kk, p+1)
            ax.imshow(patterns[p-1,:].reshape(side, side), vmin=-1, vmax=1)
            ax.set_axis_off()
    fig.canvas.draw()
    fig.savefig("patterns.png")
    plt.close(fig)

def plot_loss(losses,  session):
    # plot weights

    # plot loss

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    fig.canvas.draw()
    fig.savefig("loss.png")
    plt.close(fig)


#------------------------------------------------------------------------------
# main

def phi(x):
    # expand dimensions
    shape = x.shape
    expanded_x = broadcast(x, phi_means.shape, -1, 1)
    # distances of all x from all centroids
    dist = expanded_x - phi_means
    # radial basis
    squared_norm = tf.pow(tf.norm(dist, axis=len(dist.shape)-1),2)
    g = tf.exp(-squared_norm/(2*(phi_sigma**2))) 
    # adjust dims in case of multidimensional output
    if len(g.shape) > 1: 
        g = tf.transpose(g, [len(g.shape)-1] + list(range(len(g.shape)-1)))
    return g

def get_likelihoods(t, phi_x, W, beta):
    # mapping
    y = tf.transpose(tf.matmul(W, phi_x))
    # expand dimensions
    expanded_y = broadcast(y, t.shape, from_x=-1, from_bshape=1)
    # distances of all generated from all given data
    dist = expanded_y - t  
    ndist = tf.norm(dist, axis=len(dist.shape)-1)
    # radial basis
    A = tf.pow((beta / (2.0 * np.pi)), pattern_size/2)
    res =   tf.exp(-(beta / 2.0) * tf.pow(ndist, 2)) 
    return res, res


graph = tf.Graph()
with graph.as_default():

    x_latent = tf.placeholder(tf.float32, (None, latent_size))
    t_data = tf.placeholder(tf.float32, (None, pattern_size))
    
    phi_x = phi(x_latent)

    weights = tf.get_variable("weights", dtype=tf.float32,
            shape = (pattern_size, k_size), 
            initializer=tf.random_uniform_initializer(0, 0.001))
    beta = tf.get_variable("beta", initializer=1.)

    likelihoods, tmp = get_likelihoods(t_data, phi_x, weights, beta)

    loss = tf.reduce_sum(tf.reduce_mean(tf.log(likelihoods), axis=0))    
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    generated_patterns = tf.transpose(tf.matmul(weights, phi_x))

    with tf.Session(config=config) as session:
        
        tf.global_variables_initializer().run()
        losses = []
        
        try:
            for epoch in range(epochs):
                
                np.random.shuffle(train_data)
                # run a batch of train steps 
                elosses = []
                for batch in range(data_size//batch_size):
                     
                    curr_time = time.time() - start_time
                    if curr_time >= sim_time:
                        raise TimeOverflow("No time left!")
                   
                    # step ---- 

                    curr_batch =  train_data[batch * batch_size : (batch + 1) * batch_size ,:]
                    
                    tmp_, loss_ , _ = session.run([tmp, loss, train], 
                            feed_dict={x_latent: latents, t_data: curr_batch})
                    print(("epoch:%4d       batch:%4d    loss:%10.6f" % (epoch, batch, loss_)))
                    # --------- 
                    
                    elosses.append(loss_)
    
    
                losses.append(np.mean(elosses))
                plot_loss(losses, session)

                generated_patterns_ = session.run(generated_patterns, 
                        feed_dict={x_latent: latents})
                plot_patterns(generated_patterns_, session)
    
        except TimeOverflow:
    
            print("Plotting partial results...")
        
