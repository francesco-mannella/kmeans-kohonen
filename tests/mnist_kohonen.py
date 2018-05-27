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
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from algs.kohonen import SOM

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
batch_num = 2500
side = 28
output_channels = 20*20
epochs = 60
initial_learning_rate = 0.2

#------------------------------------------------------------------------------
# plotting 

def plots(W, losses, out_sampling, session):
    # plot weights

    W_ = W.eval()  
    minw = np.min(W_)
    maxw = np.max(W_)
    W_ = (W_- minw)/(maxw-minw)
    fig = plt.figure(figsize=(10, 10))
    kk = int(np.sqrt(output_channels))
    for i in range(kk):
        for j in range(kk):
            p = i*kk + j
            ax = fig.add_subplot(kk,kk, p+1)
            ax.imshow(W_[:,p].reshape(side, side), vmin=0, vmax=1)
            ax.set_axis_off()
    fig.canvas.draw()
    fig.savefig("weights.png")
    plt.close(fig)

    # plot loss

    plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(losses)
    fig.canvas.draw()
    fig.savefig("loss.png")
    plt.close(fig)


    fig = plt.figure(figsize=(10, 10))
    num_sampling = len(out_sampling)
    rows_sampling = num_sampling//10
    cols_sampling = 10
    for k in range(num_sampling):
        ax = fig.add_subplot(rows_sampling, cols_sampling, k+1)
        ax.imshow(out_sampling[k])
        ax.set_axis_off()
    fig.canvas.draw()
    plt.savefig("out.png")
    plt.close(fig)


#------------------------------------------------------------------------------
# main

graph = tf.Graph()
with graph.as_default():
    

    som = SOM(side*side, output_channels, batch_num, w_stddev=0.02)    

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        losses = []

        try:

            # train

            for epoch in range(epochs):
                
                np.random.shuffle(train_data)
                
                # decaying deviation 
                curr_deviation = 0.25*output_channels*np.exp(-epoch/float(epochs/6.0))
                print curr_deviation
                # decaying learning rate 
                curr_learning_rate = initial_learning_rate*np.exp(-epoch/float(epochs/6.0))

                # run a batch of train steps 
                elosses = []

                for batch in range(data_num//batch_num):
                    curr_time = time.time() - start_time
                    if curr_time >= sim_time:
                        raise TimeOverflow("No time left!")
                    
                    print "epoch:%4d       batch:%4d" % (epoch, batch)
                    
                    curr_batch =  train_data[batch * batch_num : (batch + 1) * batch_num ,:]
                    
                    _, loss_ = som.train_step(curr_batch, curr_learning_rate, 
                            curr_deviation, session)

                    elosses.append(loss_)

                losses.append(np.mean(elosses))


                if epoch % 2 == 0:
                    # test
                    
                    g_means = np.random.uniform(0, 10, [100,2]) 
                    out_sampling,_ = som.generative_step(g_means, session)
                    out_sampling = out_sampling.reshape(100, 28, 28)
                    plots(som.W, losses, out_sampling, session)


        except TimeOverflow:
            
            print "Plotting partial results..."

