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

    def __init__(self, *args, **kargs):
        

        self.rew = tf.placeholder(tf.float32, [None, 1], name="reward")

        super(RewSOM, self).__init__(*args, **kargs)

    def modulate_neighbour(self):

        modulated_phi_rk = super(RewSOM, self).modulate_neighbour()
        return tf.multiply(self.rew, modulated_phi_rk, name="mod_phi_rk") 


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

