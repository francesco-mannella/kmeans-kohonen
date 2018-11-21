# -*- coding: utf-8 -*-

### imports
#import matplotlib
#matplotlib.use("Agg")

import shutil
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
from algs.kohonen import SOM
from mpl_toolkits.mplot3d import Axes3D

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

#------------------------------------------------------------------------------
# dataset

x = np.linspace(0.2, 0.8, 2)
X, Y, Z = np.meshgrid(x, x, x)
cols = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

data = np.vstack([ 0*cols + np.random.rand(*cols.shape) 
    for i in range(50)])

#------------------------------------------------------------------------------
# parameters

data_num, input_channels = data.shape
batch_num = data_num//10
side = 50
output_channels = side*side
epochs = 300
initial_learning_rate = 0.9

plt.ion()
#------------------------------------------------------------------------------
# main

graph = tf.Graph()
with graph.as_default():
    

    som = SOM(input_channels, output_channels, batch_num, w_stddev=0.02)    

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        losses = []

        try:

            # train

            for epoch in range(epochs):
                
                np.random.shuffle(data)
                
                # decaying deviation 
                curr_deviation = 0.5*output_channels*np.exp(-epoch/float(epochs/10.0))
                # decaying learning rate 
                curr_learning_rate = initial_learning_rate*np.exp(-epoch/float(epochs/8.0))

                # run a batch of train steps 
                elosses = []

                for batch in range(data_num//batch_num):
                    curr_time = time.time() - start_time
                    if curr_time >= sim_time:
                        raise TimeOverflow("No time left!")
                    
                    
                    curr_batch =  data[batch * batch_num : (batch + 1) * batch_num ,:]
                    
                    _, loss_ = som.train_step(curr_batch, curr_learning_rate, 
                            curr_deviation, session)

                    elosses.append(loss_)

                losses.append(np.mean(elosses))
                print losses[-1]

            W = som.W.eval()

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(131, aspect="equal")
            ax.imshow(W.T.reshape(side, side, 3))
            ax1 = fig.add_subplot(133, projection='3d', aspect="equal")
            ax1.scatter(*W, c = W.T)
            ax2 = fig.add_subplot(132, projection='3d', aspect="equal")
            ax2.scatter(*data.T, c = data)
            
            # rotate the axes and update
            peak = np.hstack((np.linspace(100, 0, 31),
                np.linspace(1, 101, 31)))
            angle = np.linspace(0, 360, 62)
            for i in range(62*3):
                ax1.view_init(peak[i%62], angle[i%62])
                fig.savefig("som-%05d.png"%i)

        except TimeOverflow:
            
            print "Plotting partial results..."

