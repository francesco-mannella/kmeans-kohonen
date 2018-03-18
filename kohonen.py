### imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#-------------------------------------------------------------------------------
# only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#-------------------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=3, linewidth=999)
#-------------------------------------------------------------------------------
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
#-------------------------------------------------------------------------------


data_num = len(train_data)
batch_num = 100
side = 28
k = 64
epochs = 30
learning_rate = 0.001


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
    num_batch = win_centroids.get_shape().as_list()[0]
    win_centroids = tf.reshape(win_centroids, [1, num_batch])
    win_centroids = tf.cast(win_centroids, tf.float32)
    x = tf.stack([tf.range(num_centroids, dtype=tf.float32) for k in range(num_batch)], axis=1)

    return tf.transpose(tf.exp(-tf.pow(x - win_centroids, 2) / (2.0*tf.pow(stds, 2))))


# init a grid of sqrt(k)xsqrt(k) plots
plt.ion()
fig = plt.figure(figsize=(10, 10))
plots = []
kk = int(np.sqrt(k))
for i in range(kk):
    for j in range(kk):
        p = i*kk + j
        ax = fig.add_subplot(kk,kk, p+1)
        img = ax.imshow(np.zeros((2,2)), vmin=0, vmax=1)
        ax.set_axis_off()
        plots.append(img)

plt.show()

print len(plots)

graph = tf.Graph()
with graph.as_default():
    
    # input vector and weights 
    x = tf.placeholder(tf.float32, (batch_num, side*side))
    deviation = tf.placeholder(tf.float32, ())
    W = tf.get_variable("W", (side*side, k), 
            initializer=tf.random_normal_initializer(stddev=0.05))
    
    # broadcasting x(n,m)->(n,m,k) and W(m,k)->(n,m,k)
    xrep = tf.stack([x for j in range(k)], 2) 
    wrep = tf.stack([W for j in range(batch_num)], 0) 
    
    # distances of the side*side inputs to the side*side weights
    o = xrep - wrep
    norms =tf.norm(o, axis=1)

    # for each pattern a vectro indicating a gaussian around the winner prototipe
    wta = tf.argmin(norms, axis=1)
    wta = get_neighb(k, wta, deviation)
    
    print wta.get_shape()

    # the cost function is the summ of the distances from the winner prototipes
    loss = tf.reduce_sum(tf.multiply(tf.pow(norms, 1), wta))
    # gradient descent
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        losses = []
        for epoch in range(epochs):
            
            # decaying deviation - at each batch deviation is lower
            curr_deviation = (k/4.0)*np.exp(-epoch/float(epochs/4.0))
            
            # run a batch of train steps 
            elosses = []
            for batch in range(data_num//batch_num):
                print epoch, batch
                loss_, _ = session.run([loss, train],
                        feed_dict={
                            x: train_data[batch * batch_num : (batch + 1) * batch_num ,:],
                            deviation: curr_deviation})
                elosses.append(loss_)
            losses.append(np.mean(elosses))
                         
            # plot current weights
            W_ = W.eval()    
            for p in range(k):
                plots[p].set_data(W_[:,p].reshape(side, side))
            plt.pause(0.01)

    plt.figure()
    plt.plot(losses)
    raw_input()
                
