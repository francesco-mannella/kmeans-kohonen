# -*- coding: utf-8 -*-

### imports
import matplotlib
matplotlib.use("Agg")
import shutil
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neighborhood import get_neighb

np.set_printoptions(suppress=True, precision=3, linewidth=999)

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
shutil.rmtree("MNIST-data")

#------------------------------------------------------------------------------
# parameters
data_num = len(train_data)
batch_num = 10000
side = 28
output_channels = 64
epochs = 1002
initial_learning_rate = 0.01

#------------------------------------------------------------------------------
# main

graph = tf.Graph()
with graph.as_default():
    
    # input vector and weights 
    x = tf.placeholder(tf.float32, (batch_num, side*side))
    # deviation of the neighborhood
    deviation = tf.placeholder(tf.float32, ())
    # learning_rate
    learning_rate = tf.placeholder(tf.float32, ())
    # weights
    W = tf.get_variable("W", (side*side, output_channels), 
            initializer=tf.random_normal_initializer(stddev=0.05))
    
    # broadcasting x(n,m)->(n,m,output_channels) and W(m,output_channels)->(n,m,output_channels)
    xrep = tf.stack([x for j in range(output_channels)], 2) 
    wrep = tf.stack([W for j in range(batch_num)], 0) 
    
    # distances of the side*side inputs to the side*side weights
    o = xrep - wrep
    norms =tf.norm(o, axis=1)

    # for each pattern a vectro indicating a gaussian around the winner prototipe
    wta = tf.argmin(norms, axis=1)
    wta = get_neighb(output_channels, wta, deviation)
    
    # the cost function is the summ of the distances from the winner prototipes
    loss = tf.reduce_sum(tf.multiply(tf.pow(norms, 1), wta))
    # gradient descent
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        losses = []

        try:
            for epoch in range(epochs):
                
                # decaying deviation 
                curr_deviation = (output_channels/4.0)*np.exp(-epoch/float(epochs/4.0))
                # decaying learning rate 
                curr_learning_rate = initial_learning_rate*np.exp(-epoch/float(epochs/4.0))
            
                # run a batch of train steps 
                elosses = []
                for batch in range(data_num//batch_num):

                    curr_time = time.time() - start_time
                    if curr_time >= sim_time:
                        raise TimeOverflow("No time left!")
                    
                    print epoch, batch
                    loss_, _ = session.run([loss, train],
                            feed_dict={
                                x: train_data[batch * batch_num : (batch + 1) * batch_num ,:],
                                deviation: curr_deviation, learning_rate: curr_learning_rate})
                    elosses.append(loss_)

                losses.append(np.mean(elosses))
        
        except TimeOverflow:
            
            print "Plotting partial results..."

        # plot weights

        W_ = W.eval()  
        minw = np.min(W_)
        maxw = np.max(W_)
        W_ = W_/(maxw-minw) - minw
        fig = plt.figure(figsize=(10, 10))
        kk = int(np.sqrt(output_channels))
        for i in range(kk):
            for j in range(kk):
                p = i*kk + j
                ax = fig.add_subplot(kk,kk, p+1)
                ax.imshow(W_[:,p].reshape(side, side), vmin=0, vmax=1)
                ax.set_axis_off()
        fig.canvas.draw()
        plt.savefig("weights.png")

        # plot loss

        plt.figure()
        plt.plot(losses)
        fig.canvas.draw()
        plt.savefig("loss.png")
            
