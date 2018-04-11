# -*- coding: utf-8 -*-

### imports
import numpy as np
import tensorflow as tf


#------------------------------------------------------------------------------
# utils

def tf_flatten(x):
    dim = tf.reduce_prod(x.shape)
    return tf.reshape(x, (dim,))

#------------------------------------------------------------------------------
# SOM

class SOM(object):
    """
    Self Organizing map.

    Optimization Reduce the distance of the input patterns from the weights of the output units.
    Only those weights prototipes tha are currently the nearest to an input pattern (and their neighborhood)
    are recruited for optimization.

    Optimization is based on a generalization of the kmeans cost function.
    """

    def __init__(self, input_channels, output_channels, batch_num):
        """
        :param input_channels: length of input patterns
        :param output_channels: length of the vector of output units
        :batch_num: number of patterns presented in a single batch
        """
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.batch_num = batch_num
        self.out_side = int(np.sqrt(output_channels))
        
        
        # input vector and weights 
        self.x = tf.placeholder(tf.float32, (batch_num, self.input_channels))
        # deviation of the neighborhood
        self.deviation = tf.placeholder(tf.float32, ())
        self.curr_deviation = 0.5
        # learning_rate
        self.learning_rate = tf.placeholder(tf.float32, ())
        # weights
        self.W = tf.get_variable("W", (self.input_channels, output_channels), 
            initializer=tf.random_normal_initializer(stddev=0.01))
        
        self.phis = self.get_phis(self.output_channels, self.deviation)
        self.minimal_phi_dev = 0.005

        self.graph_model()

    def graph_model(self):
        """
        Build the tensorflow graph of the computations
        """
        
        # train

        # broadcasting x(n,m)->(n,m,output_channels) and 
        #   W(m,output_channels)->(n,m,output_channels)
        xrep = tf.stack([self.x for j in range(self.output_channels)], 2) 
        wrep = tf.stack([self.W for j in range(self.batch_num)], 0) 
        
        # distances of inputs to weights
        o = xrep - wrep
        norms =tf.norm(o, axis=1)

        # for each pattern a vector indicating a gaussian around the winner prototipe
        rk = tf.argmin(norms, axis=1)
 
        # the cost function is the sum of the distances from the winner prototipes
        self.loss = tf.reduce_sum(tf.multiply(tf.pow(norms, 2), tf.gather(self.phis, rk)))

        # gradient descent
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # generate
        self.out_means = tf.placeholder(tf.float32, (None,2))
        self.out_deviation = tf.placeholder(tf.float32, ()) 
        self.out = self.get_phis(self.output_channels, 
                self.minimal_phi_dev + self.out_deviation, self.out_means)

        self.x_sampled = tf.matmul(self.out, tf.transpose(self.W))
     
    def train_step(self, batch, dev , lr,  session):
        """
        A single batch of computations for optimization

        :param batch: a tensor with the current input patterns
        :param dev: current standard deviation of the neighborhood
                distribution in the output layer
        :param lr: current learning rate
        
        :returns: current loss value
        """
        
        loss_, _ = session.run([self.loss, self.train],
                feed_dict={
                    self.x: batch, 
                    self.deviation: dev, 
                    self.learning_rate: lr})
        return loss_
    
    def generative_step(self, means, session):
        """
        Generation of new patterns
        
        :params means: a tensor of points in the output domain 
            from which the patterns are generated

        :returns: a tensor of generated patterns

        """
        generated_patterns = session.run(self.x_sampled, 
                feed_dict={self.out_means: means, 
                    self.out_deviation: self.curr_deviation})
        return generated_patterns
    
    def get_phis(self, bins=100, sigma=0.1, means=None, dtype="float32"):
        """
        Builds a set of radial bases

        :param bins: how many bins the domain will be divided into 
        :param sigma: the std deviation of each radial basis
        :param means: the means of the radial bases. The number of means indicates 
                the number of bases. If None the means correspondst to the bins

        :returns: a tensor of radial bases
        """

        side = int(np.sqrt(bins))
        x = tf.lin_space(0.0, 1.0, side)
        X, Y = tf.meshgrid(x,x)
        centroids = tf.transpose(tf.stack([tf_flatten(X), tf_flatten(Y)]))
        if means is None:
            means = tf.identity(centroids)
        means_dim = means.get_shape().as_list()[0] 
        means_dim = means_dim if means_dim is not None else -1

        dist = tf.reshape(means, (means_dim, 1, 2)) - \
               tf.reshape(centroids, (1, bins, 2))
        phis = tf.exp(-((sigma*2.0)**-1)*tf.pow(tf.norm(dist, axis=2),2))
        return phis


