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
            w_mean=0.0, w_stddev=0.03, min_modulation=0.0001,
            max_modulation = 1.0, reproduct_deviation=0.7, 
            scope="new_som", optimizer=tf.train.AdamOptimizer ):
        """
        :param input_channels: length of input patterns
        :param output_channels: length of the vector of output units
        :param batch_num: number of patterns presented in a single batch
        :param w_stdder: standard deviation og initial weight distribution
        :param min_modulation: minimal modulation (input modulation is zero)
        :param max_modulation: maximal modulation 
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
            self.reproduct_deviation = reproduct_deviation
            self.min_modulation = min_modulation
            self.max_modulation = max_modulation
            self.optimizer = optimizer

            # weights
            self.W = tf.get_variable("som_W", (self.input_channels, 
                self.output_channels), initializer=
                tf.random_normal_initializer(mean=self.w_mean, 
                    stddev=self.w_stddev))
            
            # centroid grid in the output space 
            self.centroids = tf.get_variable(name="centroids", 
                    shape=(self.output_channels, 2), dtype=tf.float32)
            x = np.arange(self.output_side, dtype="float32")
            Y, X = np.meshgrid(x,x)
            self.centroids = tf.constant(np.vstack([X.ravel(), 
                Y.ravel()]).T, name="centroids")
            
            # TODO: substitute get_phis with interp and simplify class erasing
            #       get_phis
            self.prototype_outputs = self.get_phis(self.centroids, 
                    self.reproduct_deviation)
            
    def spreading_graph(self, x):
        """
            :param x: 2D Tensor (dtype=tf.float32, shape=(batch_num, 
                        self.input_channels)
        """
        
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

            # real outputs
            outs = tf.transpose(tf.stack((rk/self.output_side, 
                rk%self.output_side)))
            outs = tf.cast(outs, tf.float32)
            self.out_psis = interp(self.centroids, outs,
                    self.reproduct_deviation, name="out_psis")

        return self.norms, rk, self.out_psis
    
    def modulation_graph(self, modulation, rk):
        """
            :param modulation: 2D Tensor (dtype=tf.float32, 
                                (1, self.output_channels)) 
            :param rk: 1D Tensor (dtype=tf.float32, shape=(batch_num,))
        """
        with tf.variable_scope(self.scope):

            # local and global lack of modulation
            self.modulation_mean = tf.reduce_mean(modulation, 
                    name="modulation_mean")
            
            # radial bases of the output layer based on modulation measure
            self.modulation_phis = self.get_phis(self.centroids, 
                    sigma=(self.max_modulation*modulation + 
                        self.min_modulation))
            
            # The overall learning rate is linked to the modulation mean
            #self.modulation_phis = self.modulation_mean*self.modulation_phis
            
            # do the modulation
            self.neighbour = tf.gather(self.modulation_phis, rk,
                    name="neighbour")
            self.phi_rk = self.modulate_neighbour(modulation)

        return self.phi_rk

    def training_graph(self, norms, modulated_rk, learning_rate):
        """
            :param norms: 2D Tensor (dtype=tf.float32, shape=(batch_num,
                            output_channels))
            :param modulated_rk: 1D Tensor (dtype=tf.float32, 
                            shape=(batch_num, output_channels))
            :param learning_rate: 0D Tensor (dtype=tf.float32)
        """

        with tf.variable_scope(self.scope):
            
            # the cost function is the sum of the distances from
            #   the winner prototipes plus a further context modulation
            self.loss = tf.reduce_sum(tf.multiply(tf.pow(norms, 2),  
                modulated_rk), name="loss")
            # gradient descent
            self.train = self.optimizer(
                    learning_rate).minimize(
                    self.loss, var_list=[self.W], name="Gradient")

        return self.loss, self.train
    
    def get_reproduction_phis(self, means, current_dev=None, standardize=True):
        
        with tf.variable_scope(self.scope):

            if current_dev is None:

                # the resulting radial bases of the output
                reproduction_psis = interp(self.centroids, means, 
                        self.reproduct_deviation, name="reproduction_psis",
                        standardize=standardize)
            else:
                # the resulting radial bases of the output
                reproduction_psis = interp(self.centroids, means, 
                        current_dev, name="reproduction_psis", 
                        standardize=standardize)
        
        return reproduction_psis

    def reproduction_graph(self, means, current_dev=None):
        """
            :param means: 2D Tensor (dtype=tf.float32, shape=(None, 2))
        """

        self.reproduction_psis = self.get_reproduction_phis(means, current_dev)

        with tf.variable_scope(self.scope):
        
            # backprop radial bases to get the generated patterns
            self.x_sampled = tf.matmul(self.reproduction_psis,
                    self.W, transpose_b=True) 
        
        return self.x_sampled, self.reproduction_psis


    def modulate_neighbour(self, modulation):
        
        return tf.multiply(self.neighbour, modulation, name="phi_rk") 
    
    def get_phis(self, values, sigma=0.1):
        """
        Builds a set of radial bases

        :param values: the values whose distance from the grid points 
            is computed by the radial bases. 
        :param sigma: the std deviation of each radial basis

        :returns: a tensor of radial bases
        """
       
        values_dim = values.get_shape().as_list()[0] 
        values_dim = values_dim if values_dim is not None else -1

        # compute distances from grid
        dist = tf.reshape(values, (values_dim, 1, 2)) - \
            tf.reshape(self.centroids, (1, self.output_channels, 2))
        phis = tf.exp(-0.5*(sigma**-2)*tf.pow(tf.norm(dist, axis=2),2))

        return phis
    
    def generate_closed_graph(self):

        with tf.variable_scope(self.scope):

            self.x = tf.placeholder(dtype=tf.float32, shape=(self.batch_num, 
                self.input_channels), name="x")
            self.modulation = tf.placeholder(dtype=tf.float32, 
                    shape=(1, self.output_channels), name="modulation") 
            self.learning_rate =tf.placeholder(dtype=tf.float32, shape=())
            
            self.reproduction_means = tf.placeholder(dtype=tf.float32, 
                    shape=(None, 2))

        norms, rk, _ = self.spreading_graph(self.x)
        modulated_phi_rk = self.modulation_graph(self.modulation, rk)
        self.training_graph(norms, modulated_phi_rk, self.learning_rate)
        self.reproduction_graph(self.reproduction_means)

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

        norms_, y_ ,loss_, _ = session.run([self.norms, self.out_psis, 
                self.loss, self.train],
                feed_dict={
                    self.x: batch, 
                    self.learning_rate: lr,
                    self.modulation: modulation})

        return norms_, y_, loss_

    def generative_step(self, means, session):
        """
        Generation of new patterns
        
        :params means: a tensor of points in the output domain 
            from which the outputs are generated
        :param session: tensorflow session

        :returns: tensor of generated patterns and reproducttion phis

        """
        generated_patterns, psis = session.run([self.x_sampled, 
            self.reproduction_psis], feed_dict={self.gen_means: means})

        return generated_patterns, psis



