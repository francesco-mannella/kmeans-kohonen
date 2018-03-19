# -*- coding: utf-8 -*-
import tensorflow as tf

def get_neighb(num_centroids, win_centroids, stds=1.0):
    """
    Build the neighborhood gaussians around the winner units in the
    space of the output layer

    Args:
        num_centroids: number of prototipes - output units
        win_centroids: a vector with the indices of the winner centroids for the batch set of patterns 
        stds: standard deviation around the centroids

    Returns:
        a matrix where each row has a gaussian activation aaround the corrensponding winner centroid

    """

    # expand means to broadcast
    shape = win_centroids.get_shape()
    dims = len(shape)  
    wcent = tf.expand_dims(tf.cast(win_centroids, tf.float32), dims) 
    win_centroids = tf.tensordot( 
            wcent,
            tf.ones([1, num_centroids], dtype=tf.float32),
            [[dims], [0]])
    # x axis for gaussian
    x = tf.tensordot( 
            wcent*0 + 1.0, # we need a ones tensor with unknown dimension
            tf.expand_dims(tf.range(num_centroids, dtype=tf.float32), 0),
            [[dims], [0]])

    return tf.exp(-tf.pow(x - win_centroids, 2) / (2.0*tf.pow(stds, 2)))

    
                
