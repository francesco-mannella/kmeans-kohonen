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
# point dataset

train_data = np.random.uniform(-2, 2,  [10000, 2]) 

data_len, img_len = train_data.shape

labels = np.zeros(data_len)
x =  train_data[:, 0]
y =  train_data[:, 1]
labels[np.logical_and(
    ((-2 <= x) & (x < 0)),
    ((-2 <= y) & (y < 0)))] = 0
labels[np.logical_and(
    ((0 <= x) & (x <= 2)),
    ((-2 <= x) & (y < 0)))] = 1
labels[np.logical_and(
    ((-2 <= x) & (x < 0)),
    ((0 <= y) & (y <= 2)))] = 2
labels[np.logical_and(
    ((0 <= x) & (x <= 2)),
    ((0 <= y) & (y <= 2)))] = 3

labels = labels.astype(int)

data_idcs = np.arange(data_len)

#-------------------------------------------------------------------------------
# Prepare graphics

fig_weights = plt.figure(figsize=(10, 10))
ax = fig_weights.add_subplot(111)
ps = np.zeros([100, 2])
im_weights = ax.scatter(ps[:,0], ps[:,1], 
        c=plt.cm.rainbow(np.arange(100)/100.))
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

#-------------------------------------------------------------------------------

input_num = img_len
output_num = 100
output_side = int(np.sqrt(output_num))
label_num = 4
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
        [1, 1],
        [1, 8],
        [8, 1],
        [8, 8]  ])

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
                points.reshape(label_num, 1, 2), axis=-1), 
                deviation*np.exp(-epoch/(epochs*decay)))


            for batch in range(batch_len):
                
                curr_idcs = data_idcs[batch_num*batch: batch_num*(batch +1)]
                curr_train_data = train_data[curr_idcs]
                curr_labels = labels[curr_idcs]
                curr_bases = curr_radialbases[curr_labels]


                norms, outs, loss_ = ssom.train_step(
                        curr_train_data, 
                        som_learning_rate*np.exp(-epoch/(epochs*0.01)),
                        curr_bases,
                        session)             
                
            W = ssom.W.eval()
            im_weights.set_offsets(W.T)
            fig_weights.savefig("frames/w%08d.png" % epoch)

