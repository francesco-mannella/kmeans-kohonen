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
    Modulated Self Organizing Map.

    Optimization is based on a generalization of the kmeans cost function.

    Optimization Reduces the distance of the input patterns from the weights of
    the output units. Only those weights prototipes tha are currently the
    nearest to an input pattern (and their neighborhood) are recruited for
    optimization.

    Optimizations of the prototypes is also modulated globally by a given measure of 
    the geneal performance of the map and locally by a measure of the performance of
    the single prototype. 

    """

    def __init__(self, input_channels, output_channels, batch_num, 
            w_stddev=10.0, scope="new_som"):
        """
        :param input_channels: length of input patterns
        :param output_channels: length of the vector of output units
        :batch_num: number of patterns presented in a single batch
        """
        
        self.scope = scope

        with tf.variable_scope(scope):

            self.input_channels = input_channels
            self.output_channels = output_channels
            self.batch_num = batch_num
            self.output_side = int(np.sqrt(output_channels))
            
            
            # Build the tensorflow graph of the computations
            
            # main variables
            
            # input vector and weights 
            self.x = tf.placeholder(tf.float32, (batch_num, self.input_channels))
            # learning_rate
            self.learning_rate = tf.placeholder(tf.float32, ())
            # weights
            self.W = tf.get_variable("som_W", (self.input_channels, output_channels), 
                initializer=tf.random_normal_initializer(stddev=w_stddev))
            # performance for modulation
            self.perf = tf.placeholder(tf.float32, [1, self.output_channels])
            # radial bases of the output layer based on performance measure
            self.phis = self.get_phis(sigma=(1 - self.perf))

            # train graph

            # broadcasting x(n,m)->(n,m,output_channels) and 
            #   W(m,output_channels)->(n,m,output_channels)
            xrep = tf.stack([self.x for j in range(self.output_channels)], 2) 
            wrep = tf.stack([self.W for j in range(self.batch_num)], 0) 
            # local and global lack of performance
            self.uncertainty = 1.0 - self.perf
            self.uncert_mean = tf.reduce_mean(self.uncertainty)
            # distances of inputs to weights
            o = xrep - wrep
            self.norms = tf.norm(o, axis=1)
            # for each pattern a vector indicating a gaussian
            # around the winner prototipe
            rk = tf.argmin(self.norms, axis=1)
            phi_rk = tf.gather(tf.multiply(self.phis, 
                self.uncertainty) * self.uncert_mean, rk)
            # the cost function is the sum of the distances from
            # the winner prototipes plus a further context
            # modulation
            self.loss = tf.reduce_sum(tf.multiply(tf.pow(self.norms, 2), 
                tf.multiply(phi_rk, self.uncertainty)))
            # gradient descent
            self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss, var_list=[self.W])


            # generative graph

            # placeholders to get the means and deviation to
            # generate from the output layer 
            self.gen_means = tf.placeholder(tf.float32, (None,2))
            self.gen_deviation = tf.placeholder(tf.float32, ()) 
            # the initial value of the std deviation of the
            # output layer means for generation
            self.gen_default_deviation = .7
            # the resulting radial bases of the output
            self.gen_phis = self.get_phis(self.gen_means, self.gen_deviation)
            # normalized radial bases
            phi_sum = tf.reduce_sum(self.gen_phis, axis=1)
            self.gen_phis = tf.where(
                    tf.not_equal(phi_sum, 0),
                    tf.divide(self.gen_phis, tf.expand_dims(phi_sum, 1)),
                    self.gen_phis*0 + 1.0/float(self.output_channels)
                    )
            #TODO add case where phi_sum is zero
            # backprop radial bases to get the generated patterns
            self.x_sampled = tf.matmul(self.gen_phis, self.W, transpose_b=True)
            # weighted sum  of radial bases to get the patterns stddevs
            self.x_sampled_dev = tf.matmul(self.gen_phis,
                    self.perf*self.output_side, transpose_b=True)
     
    def train_step(self, batch, lr, perf, session):
        """
        A single batch of computations for optimization

        :param batch: a tensor with the current input patterns
        :param lr: current learning rate
        :param perf: an external performance surface
        
        :returns: current loss value
        """
        
        norms_, loss_, _ = session.run([self.norms, self.loss, self.train],
                feed_dict={
                    self.x: batch, 
                    self.learning_rate: lr,
                    self.perf: perf})
        return norms_, loss_
    
    def generative_step(self, means, session):
        """
        Generation of new patterns
        
        :params means: a tensor of points in the output domain 
            from which the patterns are generated

        :returns: a tensor of generated patterns

        """

        generated_patterns = session.run(self.x_sampled, feed_dict={
            self.gen_means: means, self.gen_deviation: self.gen_default_deviation})

        return generated_patterns

    def generative_devs_step(self, means, perf, session):
        """
        Generation of new patterns with standard deviations given by
        an external performance surface
        
        :params means: a tensor of points in the output domain 
            from which the patterns are generated
        :param perf: an external performance surface

        :returns: a tensor of generated patterns

        """

        generated_patterns, generated_patterns_devs, generated_phis = \
                session.run([self.x_sampled, self.x_sampled_dev, self.gen_phis], 
                feed_dict={
                    self.gen_means: means, 
                    self.gen_deviation: self.gen_default_deviation,
                    self.perf: perf})

        return generated_patterns, generated_patterns_devs, generated_phis
    
    def get_phis(self, means=None, sigma=0.1, dtype="float32"):
        """
        Builds a set of radial bases

        :param means: the means of the radial bases. The number of means indicates 
                the number of bases. If None the means correspondst to the bins
        :param sigma: the std deviation of each radial basis

        :returns: a tensor of radial bases
        """

        x = tf.range(self.output_side, dtype=tf.float32)
        Y, X = tf.meshgrid(x,x)
        centroids = tf.transpose(tf.stack([tf_flatten(X), tf_flatten(Y)]))
        if means is None:
            means = tf.identity(centroids)
        means_dim = means.get_shape().as_list()[0] 
        means_dim = means_dim if means_dim is not None else -1

        dist = tf.reshape(means, (means_dim, 1, 2)) - \
            tf.reshape(centroids, (1, self.output_channels, 2))
        phis = tf.exp(-0.5*(sigma**-2)*tf.pow(tf.norm(dist, axis=2),2))
        return phis


