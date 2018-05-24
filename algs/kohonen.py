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
    Self Organizing Map.

    Optimization is based on a generalization of the kmeans cost function.

    Optimization Reduces the distance of the input patterns from the weights of
    the output units.  Only those weights prototipes tha are currently the
    nearest to an input pattern (and their neighborhood) are recruited for
    optiization.

    """

    def __init__(self, input_channels, output_channels, batch_num, 
            w_stddev=10.0, scope="new_som"):
        """
        :param input_channels: length of input patterns
        :param output_channels: length of the vector of output units
        :param batch_num: number of patterns presented in a single batch
        :param w_stdder: standard deviation og initial weight distribution
        :param scope: the scope which the object is put into
        """
        
        self.scope = scope

        with tf.variable_scope(scope):

            self.input_channels = input_channels
            self.output_channels = output_channels
            self.batch_num = batch_num
            self.output_side = int(np.sqrt(output_channels))
            
            # centroid grid in the output space 
            x = tf.range(self.output_side, dtype=tf.float32)
            Y, X = tf.meshgrid(x,x)
            self.centroids = tf.transpose(tf.stack([tf_flatten(X), tf_flatten(Y)]))
            
            # Build the tensorflow graph of the computations
            
            # main variables
            
            # input vector and weights 
            self.x = tf.placeholder(tf.float32, (batch_num, self.input_channels))
            # deviation of the neighborhood
            self.deviation = tf.placeholder(tf.float32, ())
            # learning_rate
            self.learning_rate = tf.placeholder(tf.float32, ())
            # weights
            self.W = tf.get_variable("som_W", (self.input_channels, output_channels), 
                initializer=tf.random_normal_initializer(stddev=w_stddev))

            # train graph

            # radial bases of the output layer based on modulationormance measure
            self.phis = self.get_phis(sigma=(self.deviation), normalize=False)
            # broadcasting x(n,m)->(n,m,output_channels) and 
            #   W(m,output_channels)->(n,m,output_channels)
            xrep = tf.stack([self.x for j in range(self.output_channels)], 2) 
            wrep = tf.stack([self.W for j in range(self.batch_num)], 0) 
            # distances of inputs to weights
            o = xrep - wrep
            self.norms = tf.norm(o, axis=1)
            # for each pattern a vector indicating a gaussian around the winner prototipe
            rk = tf.argmin(self.norms, axis=1)
            rk_phis =  tf.gather(self.phis, rk)
            # the cost function is the sum of the distances from the winner prototipes
            self.loss = tf.reduce_sum(tf.multiply(tf.pow(self.norms, 2), rk_phis))
            # gradient descent
            self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            # generative graph

            # placeholders to get the means and deviation to
            # generate from the output layer 
            self.gen_means = tf.placeholder(tf.float32, (None,2))
            # the initial value of the std deviation of the
            # output layer means for generation
            self.reproduct_deviation = .7
            # the resulting radial bases of the output
            self.reproduction_phis = self.get_phis(self.gen_means, self.reproduct_deviation)
            # backprop radial bases to get the generated patterns
            self.x_sampled = tf.matmul(self.reproduction_phis, self.W, transpose_b=True)
            
    def train_step(self, batch, lr, dev, session):
        """
        A single batch of computations for optimization

        :param batch: a tensor with the current input patterns
        :param lr: current learning rate
        :param dev: current standard deviation of the neighborhood
                distribution in the output layer
        :param session: tensorflow session
        
        :returns: current distances from prototypes and loss value
        """
        
        norms_, loss_, _ = session.run([self.norms, self.loss, self.train],
                feed_dict={
                    self.x: batch, 
                    self.deviation: dev, 
                    self.learning_rate: lr})

        return norms_, loss_

    def generative_step(self, means, session):
        """
        Generation of new patterns
        
        :params means: a tensor of points in the output domain 
            from which the outputs are generated
        :param session: tensorflow session

        :returns: tensor of generated patterns and reproducttion phis

        """

        generated_patterns, reproduction_phis = session.run([self.x_sampled, 
            self.reproduction_phis], feed_dict={self.gen_means: means})

        return generated_patterns, reproduction_phis
    
    def get_phis(self, means=None, sigma=0.1, normalize=True, dtype="float32"):
        """
        Builds a set of radial bases

        :param means: the means of the radial bases. The number of means indicates 
                the number of bases. If None the means correspondst to the bins
        :param sigma: the std deviation of each radial basis

        :returns: a tensor of radial bases
        """
       
        # compute distances from grid
        if means is None:
            means = tf.identity(self.centroids)
        means_dim = means.get_shape().as_list()[0] 
        means_dim = means_dim if means_dim is not None else -1

        dist = tf.reshape(means, (means_dim, 1, 2)) - \
            tf.reshape(self.centroids, (1, self.output_channels, 2))
        phis = tf.exp(-0.5*(sigma**-2)*tf.pow(tf.norm(dist, axis=2),2))

        if normalize == True:
            # normalized radial bases
            phi_sum = tf.reduce_sum(phis, axis=1)
            phi_sum_exp = tf.expand_dims(phi_sum, axis=1)
            phis = tf.where(tf.not_equal(phi_sum, 0),
                    tf.divide(phis, phi_sum_exp),
                    tf.ones_like(phis)/float(self.output_channels))

        return phis


