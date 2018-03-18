# -*- coding: utf-8 -*-

### imports
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import shutil
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
shutil.rmtree("MNIST-data")

data_num = len(train_data)
batch_num = 10000
side = 28
filter_side = 5
input_channels = 1
output_channels = 16
epochs = 50
learning_rate = 0.01
train_data = train_data.reshape(data_num, side, side, 1)
test_data = train_data.copy()
np.random.shuffle(test_data)



graph = tf.Graph()
with graph.as_default():
    
    # input vector 
    x = tf.placeholder(tf.float32, (None, side, side, 1))
    # tensor of weight filters
    W = tf.get_variable("W", (filter_side, filter_side, input_channels, output_channels), 
            initializer=tf.random_normal_initializer(stddev=0.002))
  
    # Each filter in the convolution can be viewed as a prototipe to optimize so that its
    # euclidean distance with the patches is minimized. 
    # to easy the computations we use thevregularized weighted sum as a measure that is 
    # inversely proportional to the euclidean distance    w*patch - 0.5*trace(w*w^T) 
    weighted_sums = tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding="SAME") 
    wtraces = tf.map_fn(lambda w:  tf.trace(tf.matmul(w, tf.transpose(w))),  tf.transpose(tf.squeeze(W), [2, 0, 1]) )
    norms = weighted_sums - 0.5*tf.reshape(wtraces, [1, 1, 1, output_channels])

    # maximixe the sum of the weighted sums of the winning protoptipe for each patch in each input
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
        fig = plt.figure(figsize=(10, 10))
        kk = int(np.sqrt(output_channels))
        for i in range(kk):
            for j in range(kk):
                p = i*kk + j
                ax = fig.add_subplot(kk,kk, p+1)
                img = ax.imshow(W_[:,:,0,p], vmin=0, vmax=1)
                ax.set_axis_off()
        plt.savefig("weights.png")

        n_tests = 3
        weighted_sums_ = session.run(weighted_sums, feed_dict={x: test_data[:n_tests]})
        mino = np.min(weighted_sums_)
        maxo = np.max(weighted_sums_)
        weighted_sums_ = weighted_sums_/(maxo-mino) - mino
        fig = plt.figure(figsize=(16, 6))
        for q in range(n_tests):
            for i in range(output_channels): 
                    p = 2*(output_channels*q + i)
                    ax1 = fig.add_subplot(2*n_tests, output_channels, p+1)
                    ax1.imshow(test_data[q,:,:,0], vmin=0, vmax=1)
                    ax1.set_axis_off()
                    ax2 = fig.add_subplot(2*n_tests, output_channels, p+2)
                    ax2.imshow(weighted_sums_[q,:,:,i], vmin=0, vmax=1)
                    ax2.set_axis_off()
        plt.savefig("weighted_sums")

       
        plt.figure()
        plt.plot(losses)
        plt.savefig("loss.png")


        


    
                