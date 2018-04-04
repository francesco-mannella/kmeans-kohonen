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
import neighborhood
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
batch_num = 1000
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


    fig = plt.figure()
    num_sampling = len(out_sampling)
    rows_sampling = num_sampling//10
    cols_sampling = 10
    for k in range(num_sampling):
        ax = fig.add_subplot(rows_sampling, cols_sampling, k+1)
        ax.imshow(out_sampling[k])
        ax.set_axis_off()
    fig.canvas.draw()
    plt.savefig("out.png")

#------------------------------------------------------------------------------
# som

class SOM(object):

    def __init__(self, input_channels, output_channels, batch_num):
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.batch_num = batch_num
        self.side = int(np.sqrt(input_channels))
        self.out_side = int(np.sqrt(output_channels))

        
        # input vector and weights 
        self.x = tf.placeholder(tf.float32, (batch_num, self.side*self.side))
        # deviation of the neighborhood
        self.deviation = tf.placeholder(tf.float32, ())
        # learning_rate
        self.learning_rate = tf.placeholder(tf.float32, ())
        # weights
        self.W = tf.get_variable("W", (self.side*self.side, output_channels), 
            initializer=tf.random_normal_initializer(stddev=0.05))

        self.graph_model()

    def graph_model(self):
        
        # train

        # broadcasting x(n,m)->(n,m,output_channels) and W(m,output_channels)->(n,m,output_channels)
        xrep = tf.stack([self.x for j in range(self.output_channels)], 2) 
        wrep = tf.stack([self.W for j in range(self.batch_num)], 0) 
        
        # distances of the side*side inputs to the side*side weights
        o = xrep - wrep
        norms =tf.norm(o, axis=1)

        # for each pattern a vectro indicating a gaussian around the winner prototipe
        rk = tf.argmin(norms, axis=1)
        rk = get_neighb(self.output_channels, rk, self.deviation, neighborhood.gauss2D) 
        
        # the cost function is the summ of the distances from the winner prototipes
        self.loss = tf.reduce_sum(tf.multiply(tf.pow(norms, 1), rk))

        # gradient descent
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # generate
        self.out_means = tf.placeholder(tf.float32, (None,))
        self.out_dev = tf.placeholder(tf.float32, ())
        out = get_neighb(self.output_channels, self.out_means, 
                self.out_dev, neighborhood.gauss2D) 
        self.x_sampled = tf.matmul(out, tf.transpose(self.W))
        self.x_sampled = tf.reshape(self.x_sampled, [-1, self.side, self.side])
        
    def train_step(self, batch, dev ,lr, session):
        
        loss_, _ = session.run([self.loss, self.train],
                feed_dict={
                    self.x: batch, 
                    self.deviation: dev, 
                    self.learning_rate: lr})
        return loss_
    
    def generative_step(self, means, dev, session):
        
        generated_patterns = session.run(self.x_sampled,
                feed_dict={self.out_means: means, self.out_dev: dev})
        out_side = int(np.sqrt(self.output_channels))
        return generated_patterns
    

#------------------------------------------------------------------------------
# main

graph = tf.Graph()
with graph.as_default():
    

    som = SOM(side*side, output_channels, batch_num)    

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        losses = []

        try:

            # train

            for epoch in range(epochs):
                
                np.random.shuffle(train_data)
                
                # decaying deviation 
                curr_deviation = int(np.sqrt(output_channels))*np.exp(-epoch/float(epochs/6.0))
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
                    
                    loss_ = som.train_step(curr_batch, curr_deviation, curr_learning_rate, session)

                    elosses.append(loss_)

                losses.append(np.mean(elosses))


            # test
            
            g_means = np.random.uniform(0,som.output_channels-1, 100) 
            g_dev = 0.1
            out_sampling = som.generative_step(g_means, g_dev, session)
            
            plots(som.W, losses, out_sampling, session)


        except TimeOverflow:
            
            print "Plotting partial results..."

