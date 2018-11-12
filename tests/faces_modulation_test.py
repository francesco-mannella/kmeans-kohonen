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
# load faces dataset

faces = np.loadtxt("../faces/figs")
images = faces 
data_len, img_len = images.shape
data_idcs = np.arange(data_len)

slabels = np.loadtxt("../faces/labels", dtype="string")
label_types = np.unique(slabels)
labels = np.zeros(data_len, dtype="int") 
for i, lab in enumerate(label_types):
    labels[slabels==lab] = i


#-------------------------------------------------------------------------------
# Prepare graphics

def reshape_weights(W):
    n_inp, n_out = W.shape
    oside = int(np.sqrt(n_out))
    w = 152
    h = 107 
    
    res = np.zeros([oside*w, oside*h])
    
    for o in range(n_out):
        ww = W[:, o].reshape(w, h)
        row = w*(o//oside) 
        col = h*(o%oside) 
        res[row:(row + w), col:(col + h)] = ww

    return res

fig_weights = plt.figure(figsize=(16, 10))
ax = fig_weights.add_subplot(121)
im_weights = ax.imshow(np.zeros([2,2]),vmin=0, vmax=255,
        cmap=plt.cm.gray)
ax.axis('off')
ax2 = fig_weights.add_subplot(122, aspect="equal")
#-------------------------------------------------------------------------------

input_num = img_len
output_num = 100
output_side = int(np.sqrt(output_num))
batch_num = 100
epochs = 200
batch_len = data_len / batch_num 
decay = 0.15

som_learning_rate = 4.0
deviation = output_side/2.0

x = np.arange(output_side)
X, Y = np.meshgrid(x, x)
grid = np.vstack([X.ravel(), Y.ravel()]).T

points = np.array([
    [1. , 1. ],
    [1. , 8. ],
    [2.5, 4.5],
    [4.5, 2.5],
    [4.5, 6.5],
    [6.5, 4.5],
    [8. , 1. ],
    [8. , 8. ]])

for i, p in enumerate(points):
    ax2.text(p[1]-.5, p[0], label_types[i])

ax2.text(11, 7, "A=ASIAN\nW=WHITE\nL=LATINO\nB=BLACK\n\nF=FEMALE\nM=MALE")
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xlim([-2, 16])
ax2.set_ylim([12, -2])

graph = tf.Graph()
with graph.as_default():

    ssom = SOM(input_num, output_num, batch_num, w_stddev=0.01, 
            neighborhood=deviation)

    ssom.generate_closed_graph()

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        

        for epoch in range(epochs):       
            
            np.random.shuffle(data_idcs)
            curr_learning_rate = som_learning_rate *\
                    np.exp(-epoch/(epochs*decay))
            curr_radialbases = gauss(np.linalg.norm(
                grid.reshape(1, output_num, 2) - 
                points.reshape(len(label_types), 1, 2), axis=-1), 
                deviation*np.exp(-epoch/(epochs*decay)))

            for batch in range(batch_len):
                
                curr_idcs = data_idcs[batch_num*batch: batch_num*(batch + 1)]
                curr_images = images[curr_idcs]
                curr_labels = labels[curr_idcs]
                curr_bases = curr_radialbases[curr_labels]

                norms, outs, loss_ = ssom.train_step(
                        curr_images, 
                        som_learning_rate*np.exp(-epoch/(epochs*decay)),
                        curr_bases,
                        session)             
                
            W = ssom.W.eval()
            im_weights.set_data(reshape_weights(W).T)
            fig_weights.savefig("frames/w%08d.png" % epoch)
    

