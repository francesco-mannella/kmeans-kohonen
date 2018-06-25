# -*- coding: utf-8 -*-

### imports
import numpy as np
import tensorflow as tf
from modulated_kohonen import SOM


#------------------------------------------------------------------------------
# RewSOM

class RewSOM(SOM):
    """
    Modulated Reward-based Self Organizing Map.
    
    Adds reward signal to a modulated SOM.

    """

    def modulation_graph(self, rew , *args, **kargs):

        modulated_phi_rk = super(RewSOM, self).modulation_graph(*args, **kargs)
        return tf.multiply(rew, modulated_phi_rk, name="rew_phi_rk") 

    def generate_closed_graph(self):

        with tf.variable_scope(self.scope):

            self.x = tf.placeholder(dtype=tf.float32, 
                    shape=(self.batch_num, self.input_channels), name="x")
            self.rew = tf.placeholder(idtype=tf.float32, 
                    shape=(None, 1), name="reward")
            self.modulation = tf.placeholder(dtype=tf.float32, 
                    shape=(1, self.output_channels), name="modulation") 
            self.learning_rate =tf.placeholder(dtype=tf.float32, shape=())
            
            self.reproduction_means = tf.placeholder(dtype=tf.float32, 
                    shape=(None, 2))

        norms, rk, _ = self.spreading_graph(self.x)
        rew_phi_rk = self.modulation_graph(self.rew, self.modulation, rk) 
        self.training_graph(norms, rew_phi_rk, self.learning_rate)
        self.reproduction_graph(self.reproduction_means)


    def train_step(self, batch, lr, modulation, rew,  session):
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
                    self.modulation: modulation,
                    self.rew: rew})

        return norms_, y_, loss_

