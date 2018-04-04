# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def gauss1D(mu, sigma, xdims): 
    
    # expand means to broadcast
    shape = mu.get_shape()
    dims = len(shape)  
    mu = tf.expand_dims(tf.cast(mu, tf.float32), dims) 
    mu_expanded = tf.tensordot( 
            mu,
            tf.ones([1, xdims], dtype=tf.float32),
            [[dims], [0]])
    # x axis for gaussian
    x = tf.tensordot( 
            mu*0 + 1.0, # we need a ones tensor with unknown dimension
            tf.expand_dims(tf.range(xdims, dtype=tf.float32), 0),
            [[dims], [0]])
    
    return tf.exp(-tf.pow(x - mu_expanded, 2) / (2.0*tf.pow(sigma, 2)))

def gauss2D(mu, sigma, xdims): 
    
    side_dims = int(np.sqrt(xdims))

    # expand means to broadcast
    shape = mu.get_shape()
    dims = len(shape)  
    mu = tf.expand_dims(tf.cast(mu, tf.float32), dims) 
    mu_expanded = tf.tensordot( 
            mu, 
            tf.ones([1, xdims], dtype=tf.float32),
            [[dims], [0]])
    #  original x axis for gaussian
    x = tf.tensordot( 
            mu*0 + 1.0, # we need a ones tensor with unknown dimension
            tf.expand_dims(tf.range(xdims, dtype=tf.float32), 0), 
            [[dims], [0]])
    
    X = x//side_dims
    Y = x%side_dims
    muX = mu_expanded // side_dims
    muY = mu_expanded % side_dims

    return  tf.exp(-(tf.pow(X - muX, 2) + tf.pow(Y - muY, 2)) / (2.0*tf.pow(sigma, 2)))

def get_neighb(num_centroids, win_centroids, stds=1.0, gauss=gauss1D):
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
 
    res = gauss(win_centroids, stds, num_centroids) 
    return res
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    session = tf.InteractiveSession()
   
    m = tf.placeholder(tf.float32, (2,))
    s = tf.placeholder(tf.float32, ())
    
    o = get_neighb(20*20, m, s, gauss2D)

    m_ = np.linspace(0, 20*20, 2)
    s_ = 1 
    o_ = session.run(o, feed_dict={m:m_, s:s_}) 

    print o_.shape
    for u in o_:
        plt.imshow(u.reshape(20,20))
        plt.show()

