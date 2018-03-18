# -*- coding: utf-8 -*-

### imports
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import time
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('-t','--time',
        help="Simulation time (sec)",
        action="store", default=3600)  
args = parser.parse_args()

sim_time = int(args.time)
start_time = time.time()

#-------------------------------------------------------------------------------
# only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#-------------------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=3, linewidth=999)
#-------------------------------------------------------------------------------
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
#-------------------------------------------------------------------------------


data_num = len(train_data)
batch_num = 10000
side = 28
filter_side = 5
input_channels = 1
output_channels = 16
epochs = 50
learning_rate = 0.01
train_data = train_data.reshape(data_num, side, side, 1)


# init a grid of sqrt(output_channels)xsqrt(output_channels) plots

fig = plt.figure(figsize=(10, 10))
plots = []
axes = []
kk = int(np.sqrt(output_channels))
for i in range(kk):
    for j in range(kk):
        p = i*kk + j
        ax = fig.add_subplot(kk,kk, p+1)
        img = ax.imshow(np.zeros((2,2)), vmin=0, vmax=1)
        ax.set_axis_off()
        plots.append(img)
        axes.append(ax)

graph = tf.Graph()
with graph.as_default():
    
    # input vector and weights 
    x = tf.placeholder(tf.float32, (batch_num, side, side, 1))
    W = tf.get_variable("W", (filter_side, filter_side, input_channels, output_channels), 
            initializer=tf.random_normal_initializer(stddev=0.002))
    out = tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding="SAME") 
    wtraces = tf.map_fn(lambda w:  tf.trace(tf.matmul(w, tf.transpose(w))),  tf.transpose(tf.squeeze(W), [2, 0, 1]) )
    norms = out - 0.5*tf.reshape(wtraces, [1, 1, 1, output_channels])
    #norm_means = tf.reduce_sum(norms, axis=[1, 2])
    rk = tf.one_hot(tf.argmax(norms, axis=3), output_channels, dtype=tf.float32)
    loss = -tf.reduce_sum(rk * norms)
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[W])

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        losses = []
        
        try:
            for epoch in range(epochs):
                
                # run a batch of train steps 
                elosses = []
                for batch in range(data_num//batch_num):
                    curr_time = time.time() - start_time
                    if curr_time >= sim_time:
                        raise "No time left!"
                    print epoch, batch
                    current_batch = train_data[batch * batch_num : (batch + 1) * batch_num ,:]
                    loss_, _ = session.run([loss, train], feed_dict={x: current_batch})
                    elosses.append(loss_)
                losses.append(np.mean(elosses))
        except:
            print "Plotting partial results..."
                
        W_ = W.eval()  
        minw = np.min(W_)
        maxw = np.max(W_)
        W_ = W_/(maxw-minw) - minw
        print np.min(W_), np.max(W_)
        for p in range(output_channels):
            plots[p].set_data(W_[:,:,0,p])
        plt.savefig("weights.png")

        plt.figure()
        plt.plot(losses)
        plt.savefig("loss.png")
    
                
