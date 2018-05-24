# -*- coding: utf-8 -*-

### imports
import numpy as np
import tensorflow as tf


#------------------------------------------------------------------------------
# utils

def tf_flatten(x):
    dim = tf.reduce_prod(x.shape)
    return tf.reshape(x, (dim,))
    
def interp(grid, means, sigma=0.1, name="psis"):
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
    phis = tf.divide(phis,tf.reshape(tf.reduce_sum(phis, axis=1), (means_num, 1)), name=name)
    return phis

#------------------------------------------------------------------------------
# SOM

class SOM(object):
    """
    Modulated Self Organizing Map.

    Optimization is based on a generalization of the kmeans cost function.

    Optimization Reduces the distance of the input patterns from the weights of
    the output units. Only those weights prototipes tha are currently the
    nearest to an input pattern (and their neighborhood) are recruited for
    optimization.

    Optimizations of the prototypes is also modulated globally by a given measure of 
    the geneal modulationormance of the map and locally by a measure of the modulation of
    the single prototype. 

    """

    def __init__(self, input_channels, output_channels, batch_num, 
            w_mean=0.0, w_stddev=10.0, min_modulation=0.7, scope="new_som"):
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
            self.min_modulation = min_modulation
            
            # centroid grid in the output space 
            x = tf.range(self.output_side, dtype=tf.float32)
            Y, X = tf.meshgrid(x,x)
            self.centroids = tf.transpose(tf.stack([tf_flatten(X), tf_flatten(Y)]), name="centroids")
            
            # Build the tensorflow graph of the computations
            
            # main variables
            
            # input vector and weights 
            self.x = tf.placeholder(tf.float32, (batch_num, self.input_channels), name="x")
            # learning_rate
            self.learning_rate = tf.placeholder(tf.float32, (), name="lr")
            # weights
            self.W = tf.get_variable("som_W", (self.input_channels, output_channels), 
                initializer=tf.random_normal_initializer(mean=w_mean, stddev=w_stddev))
            # modulation
            self.modulation = tf.placeholder(tf.float32, [1, self.output_channels], name="modulation")
            # radial bases of the output layer based on modulation measure
            self.modulation_phis = self.get_phis(self.centroids,
                    sigma=(self.modulation+self.min_modulation))
            
            # train graph
            # local and global lack of modulationormance
            self.modulation_mean = tf.reduce_mean(self.modulation, name="modulation_mean")
            # broadcasting x and W 
            xrep = tf.reshape(self.x, (batch_num, self.input_channels, 1))
            wrep = tf.reshape(self.W, (1, self.input_channels, self.output_channels))
            # distances of inputs from weights
            o = xrep - wrep
            self.norms = tf.norm(o, axis=1, name="norms")
            # for each pattern a vector indicating a gaussian around the winner prototipe
            rk = tf.argmin(self.norms, axis=1, name="rk")
            self.neighbour = tf.gather(self.modulation_phis, rk, name="neighbour")

            # do the modulation
            self.phi_rk = self.modulate_neighbour()
            
            # the cost function is the sum of the distances from
            #   the winner prototipes plus a further context modulation
            self.loss = tf.reduce_sum(tf.multiply(tf.pow(self.norms, 2),  self.phi_rk), name="loss")
            # # gradient descent
            # self.train = tf.train.AdamOptimizer(
            #         self.learning_rate*self.modulation_mean).minimize(
            #         self.loss, var_list=[self.W])
            # gradient descent
            self.train = tf.train.AdamOptimizer(
                    self.learning_rate).minimize(
                    self.loss, var_list=[self.W], name="Gradient")

            # generative graph

            # placeholders to get the means and deviation to
            # generate from the output layer 
            self.gen_means = tf.placeholder(tf.float32, (None,2), name="gen_means")
            # the initial value of the std deviation of the
            # output layer means for generation
            self.reproduct_deviation = 1.7

            # the resulting radial bases of the output
            self.reproduction_psis = interp(self.centroids, 
                    self.gen_means, self.reproduct_deviation, name="reproduction_psis")
            # backprop radial bases to get the generated patterns
            self.x_sampled = tf.matmul(self.reproduction_psis, self.W, transpose_b=True)
          
            # utils
            self.prototype_outputs = self.get_phis(self.centroids, self.reproduct_deviation)

            outs = tf.transpose(tf.stack((rk/self.output_side, rk%self.output_side)))
            outs = tf.cast(outs, tf.float32)
            # real outputs
            self.out_psis = interp(self.centroids, outs,
                    self.reproduct_deviation, name="out_psis")

    def modulate_neighbour(self):
        
        # radial bases of the output layer based on modulation measure
        self.modulation_phis = self.get_phis(self.centroids,
                sigma=(self.modulation))

        return tf.multiply(self.neighbour, self.modulation, name="phi_rk") 


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

    def inject_weights(self, weights, session):
        
        session.run(self.r_weights, feed_dict={self.weight_noise: weights})

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


