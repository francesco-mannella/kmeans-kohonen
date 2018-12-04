### imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
import time
import matplotlib
matplotlib.use("Agg")
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from algs.modulated_kohonen import SOM

#-------------------------------------------------------------------------------
# only current needed GPU memory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#-------------------------------------------------------------------------------

np.set_printoptions(suppress=True, precision=3, linewidth=999)

#-------------------------------------------------------------------------------

def gauss(d, s): return np.exp(-0.5*(s**-2)*d**2)

#-------------------------------------------------------------------------------

if not os.path.exists("frames"): os.makedirs("frames")  

#-------------------------------------------------------------------------------
# load MNIST dataset

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
shutil.rmtree("MNIST-data")

images = mnist.train.images
labels = mnist.train.labels
data_len, img_len = images.shape
data_idcs = np.arange(data_len)

#-------------------------------------------------------------------------------
# Prepare graphics

def reshape_weights(W):
    n_inp, n_out = W.shape
    oside = int(np.sqrt(n_out))
    side = int(np.sqrt(n_inp))
    
    res = np.zeros([oside*side, oside*side])
    
    for o in range(n_out):
        w = W[:, o].reshape(side, side)
        row = side*(o//oside) 
        col = side*(o%oside) 
        res[row:(row + side), col:(col + side)] = w

    return res

fig_weights = plt.figure(figsize=(8, 4))
ax = fig_weights.add_subplot(121, aspect="equal")
im_weights = ax.imshow(np.zeros([2,2]), vmin=0, vmax=1)
ax.set_axis_off()
ax1 = fig_weights.add_subplot(122, aspect="equal")
#-------------------------------------------------------------------------------

input_num = img_len
output_num = 100
output_side = int(np.sqrt(output_num))
batch_num = 500
epochs = 3000
batch_len = data_len / batch_num 
decay = 0.05

som_learning_rate = 4.0
deviation = output_side/16.0

x = np.arange(output_side)
X, Y = np.meshgrid(x, x)
grid = np.vstack([X.ravel(), Y.ravel()]).T

points = np.array(
       [
        [1, 3],
        [1, 6],
        [3, 1],
        [3, 8],
        [4.5, 3],
        [4.5, 6],
        [6, 1],
        [6, 8],
        [8, 3],
        [8, 6]])

ax1.set_xlim([-1, output_side])
ax1.set_xticks(list(range(output_side)))
ax1.set_ylim([-1, output_side])
ax1.set_yticks(list(range(output_side)))
for i, p in enumerate(points):
    ax1.add_artist(plt.Circle(p, 1.5, 
        edgecolor='black', facecolor='white' ))
    ax1.text(p[0], 9 - p[1], s="%d"%i, fontsize=30, 
            verticalalignment='center',
            horizontalalignment='center')

radial_bases = gauss(np.linalg.norm(
    grid.reshape(1, output_num, 2) - 
    points.reshape(output_side, 1, 2), axis=-1), 
    deviation)

graph = tf.Graph()
with graph.as_default():

    ssom = SOM(input_num, output_num, batch_num, w_stddev=0.01, 
            neighborhood=deviation)

    ssom.generate_closed_graph()

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        

        for epoch in range(epochs):       
            
            np.random.shuffle(data_idcs)
            curr_learning_rate = som_learning_rate*np.exp(-epoch/(epochs*decay))
            curr_radialbases =  gauss(np.linalg.norm(
                grid.reshape(1, output_num, 2) - 
                points.reshape(output_side, 1, 2), axis=-1), 
                deviation*np.exp(-epoch/(epochs*decay)))


            for batch in range(batch_len):
                
                curr_idcs = data_idcs[batch_num*batch: batch_num*(batch +1)]
                curr_images = images[curr_idcs]
                curr_labels = labels[curr_idcs]
                curr_bases = curr_radialbases[curr_labels]


                norms, outs, loss_ = ssom.train_step(
                        curr_images, 
                        som_learning_rate*np.exp(-epoch/(epochs*0.01)),
                        curr_bases,
                        session)             
                
            W = ssom.W.eval()
            im_weights.set_data(reshape_weights(W))
            fig_weights.savefig("frames/w%08d.png" % epoch)

