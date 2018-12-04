# -*- coding: utf-8 -*-

### imports
import numpy as np
import tensorflow as tf


#------------------------------------------------------------------------------
# utils

def tf_flatten(x):
    dim = tf.reduce_prod(x.shape)
    return tf.reshape(x, (dim,))
    
def interp(grid, means, sigma=0.1, name="psis", standardize=True):
    """
    Builds a set of radial bases

    :param grid: the grid of centroids filling the 2D linspace 
    :param means: the means of the radial bases. 
    :param sigma: the std deviation of each radial basis

    :returns: a tensor of radial bases
    """
    means_num = means.get_shape().as_list()[0] 
    means_num = means_num if means_num is not None else -1
    grid_num = grid.shape[0]

    # compute distances from grid
    dist = tf.reshape(means, (means_num, 1, 2)) - \
        tf.reshape(grid, (1, grid_num, 2))
   
    if (type(sigma) is not float) :
        rsp =  [1 for x in range(len(means.get_shape())-1)] +[-1]
        sigma = tf.reshape(sigma, rsp)

    phis = tf.exp(-0.5*(sigma**-2)*tf.pow(tf.norm(dist, axis=2),2))
    if standardize == True:
        phis = tf.divide(phis,tf.reshape(tf.reduce_sum(phis, axis=1), 
            (means_num, 1)), name=name)
    return phis

#------------------------------------------------------------------------------
# SOM

class SOM(object):
    """
    Modulated Self Organizing Map.

    Optimization is based on a generalization of the kmeans cost function.

    Optimization Reduces the distance of the input patterns from the
    weights of the output units. Only those weights prototipes tha are
    currently the nearest to an input pattern (and their neighborhood) are
    recruited for optimization.

    Optimizations of the prototypes is also modulated globally by a given
    measure of the geneal modulationormance of the map and locally by a
    measure of the modulation of the single prototype. 

    """

    def __init__(self, input_channels, output_channels, batch_num, 
            w_mean=0.0, w_stddev=0.03, neighborhood=0.7, 
            scope="new_som", optimizer=tf.train.AdamOptimizer ):
        """
        :param input_channels: length of input patterns
        :param output_channels: length of the vector of output units
        :param batch_num: number of patterns presented in a single batch
        :param w_stdder: standard deviation og initial weight distribution
        :param neighborhood: standard deviation of radial bases from winners
        :param scope: the scope which the object is put into
        """
        
        self.scope = scope

        with tf.variable_scope(scope):

            self.input_channels = input_channels
            self.output_channels = output_channels
            self.batch_num = batch_num
            self.output_side = int(np.sqrt(output_channels))
            self.w_mean = w_mean
            self.w_stddev = w_stddev
            self.neighborhood = neighborhood
            self.optimizer = optimizer

            # weights
            self.W = tf.get_variable("som_W", (self.input_channels, 
                self.output_channels), initializer=
                tf.random_normal_initializer(mean=self.w_mean, 
                    stddev=self.w_stddev))
            
            # centroid grid in the output space 
            x = np.arange(self.output_side, dtype="float32")
            Y, X = np.meshgrid(x,x)
            self.centroids = tf.constant(np.vstack([X.ravel(), 
                Y.ravel()]).T, name="centroids")
            
            self.neighborhood_bases = interp(self.centroids, self.centroids, 
                    self.neighborhood, "neighborhood_bases", standardize=False)
            
    
    def spreading_graph(self, x, neigh = None, return_value="bases"):
        """
            :param x: 2D Tensor (dtype=tf.float32, shape=(batch_num, 
                        self.input_channels)
            :param neigh: current neighborhood for the generated bases 
            :param return_value: {"bases", "outs"}
        """
        
        if neigh is None:
            neigh = self.neighborhood

        with tf.variable_scope(self.scope):

            # spreading 
            # broadcasting x and W 
            xrep = tf.reshape(x, (self.batch_num, self.input_channels, 1))
            wrep = tf.reshape(self.W, (1, self.input_channels, 
                self.output_channels))
            # distances of inputs from weights
            o = xrep - wrep
            self.norms = tf.norm(o, axis=1, name="norms")
            # for each pattern a vector indicating a gaussian around 
            #    the winner prototipe
            rk = tf.argmin(self.norms, axis=1, name="rk")
            rk_bases = tf.gather(self.neighborhood_bases, rk)

            # real outputs
            outs = tf.transpose(tf.stack((rk//self.output_side, 
                rk%self.output_side)))
            outs = tf.cast(outs, tf.float32)
            out_bases = interp(self.centroids, outs,
                    neigh , name="out_bases")
        if return_value == "bases":
            return self.norms, rk_bases, out_bases
        elif return_value == "outs":
            return self.norms, rk_bases, out_bases, outs

    def backpropagate_graph(self, output_points, current_dev=None):
        """
            :param output_points: 2D Tensor (dtype=tf.float32, shape=(None, 2))
        """
        
        if current_dev is None:
            current_dev = self.neighborhood

        out_bases = interp(self.centroids, output_points, 
                        current_dev, name="backward_bases",
                        standardize=True)

        with tf.variable_scope(self.scope):
        
            # backprop radial bases to get the generated patterns
            x_sampled = tf.matmul(out_bases, self.W, transpose_b=True) 
        
        return x_sampled, out_bases

    def compute_loss(self, norms, modulations):
        with tf.variable_scope(self.scope):
            
            # the cost function is the sum of the modulated input-prototyres 
            # distances
            self.loss = tf.reduce_sum(tf.multiply(tf.pow(norms, 2),  
                modulations), name="loss")
        
        return self.loss

    def training_graph(self, norms, modulations, learning_rate):
        """
            :param norms: 2D Tensor (dtype=tf.float32, shape=(batch_num,
                            output_channels))
            :param modulations: 1D Tensor (dtype=tf.float32, 
                            shape=(batch_num, output_channels))
            :param learning_rate: 0D Tensor (dtype=tf.float32)
        """

        self.compute_loss(norms, modulations)

        with tf.variable_scope(self.scope):
            
            # gradient descent
            self.train = self.optimizer(
                    learning_rate).minimize(
                    self.loss, var_list=[self.W], name="Gradient")

        return self.loss, self.train
                 
    def generate_closed_graph(self):

        with tf.variable_scope(self.scope):

            self.x = tf.placeholder(dtype=tf.float32, shape=(self.batch_num, 
                self.input_channels), name="x")
            self.modulation = tf.placeholder(dtype=tf.float32, 
                    shape=(None, self.output_channels), name="modulation") 
            self.learning_rate =tf.placeholder(dtype=tf.float32, shape=())
            
        self.norms, rk_bases, self.out_bases  = self.spreading_graph(self.x)
        self.loss, self.train = self.training_graph(self.norms, 
                rk_bases*self.modulation, self.learning_rate)
        
    def generate_custom_closed_graph(self):

        with tf.variable_scope(self.scope):

            self.x = tf.placeholder(dtype=tf.float32, shape=(self.batch_num, 
                self.input_channels), name="x")
            self.modulation = tf.placeholder(dtype=tf.float32, 
                    shape=(None, self.output_channels), name="modulation") 
            self.learning_rate =tf.placeholder(dtype=tf.float32, shape=())
            
        self.norms, rk_bases, self.out_bases  = self.spreading_graph(self.x)
        self.loss, self.train = self.training_graph(self.norms, 
                self.modulation, self.learning_rate)

    def train_step(self, batch, lr, modulation, session):
        """
        A single batch of computations for optimization

        :param batch: a tensor with the current input patterns
        :param lr: current learning rate
        :param modulation: an external modulation surface
        :param session: tensorflow session
        
        :returns: current distances from prototypes and loss value
        """

        if modulation is None:
            modulation = np.ones([1, self.output_channels])

        norms_, y_ ,loss_, _ = session.run([self.norms, self.out_bases, 
                self.loss, self.train],
                feed_dict={
                    self.x: batch, 
                    self.learning_rate: lr,
                    self.modulation: modulation})

        return norms_, y_, loss_



