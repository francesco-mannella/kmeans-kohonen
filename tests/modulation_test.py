### imports
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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

# This example shows how a SOM can be modulated. We can modulate the activity
# of the SOM so that the units in the output layer are forced to learn
# prototypes of well defined subgroups of the data. Moreover we can force which
# group of units learns the prototype for a soubgroup.  As a result two
# features emerge:
#
# 1) The topology of the output layer in the input space can be modelled at
# will. 
#
# 2) The network can learn sequentially with different parts of the dataset
# experienced at different periods of the network learning phase. 
#
# In the this particular case the inputs to the network are oints in a 2D space
# (x and y coordinate) and the 100 output units are disposed on a 10x10 2D
# grid. We sample data at each epoch from one of a set of normal distributions
# whose means are disposed on a grid in the 2D input space. Each distribution
# is linked to a different modulation centroid in the 2D output space
# determined by the indices of the output units. In particular we rotate the
# points corresponding to the output units' indices around the central index
# (4,4) so that the 10x10 modulation centroids are not overlapping with the
# output units' positions. The modulation at each unit is then computed as a
# radial basis of the distance of this chosen centroid from the position of the
# unit in the output space. The network learns to read each dataset with the
# group of units that is the nearest to the corresponding modulation centroid.
# The resulting topology of the output units in the input space is the same of 
# the one of the modulation centroids in the output space.

input_num = 2
output_num = 100
output_side = int(np.sqrt(output_num))
batch_num = 2000
epochs = 2000

som_learning_rate = 0.03
rep_deviation = 0.7

x = np.arange(float(output_side))
X, Y = np.meshgrid(x, x)
means = np.vstack((Y.ravel(), X.ravel())).T
theta = np.pi*0.25
rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
rotated = np.dot(means - output_side/2., rot)
rotated += output_side/2.

dists = np.linalg.norm(means.reshape(output_num, 1, 2) - rotated.reshape(1, output_num, 2), axis=2)
centroids = gauss(dists, rep_deviation)

#------------------------------------------------------------------------------
# main
colors=plt.cm.jet(np.arange(output_num)/float(output_num))
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, aspect="equal")
wgrid = []; rows = []; cols = []
for i in range(output_side):
    line, = ax.plot(0,0, c="red", alpha=2.2)
    rows.append(line)
for j in range(output_side):
    line, = ax.plot(0,0, alpha=0.5)
    cols.append(line)
wgrid.append(rows); wgrid.append(cols)
wiscatter = ax.scatter(0,0,c="grey", s=100)
wscatter = ax.scatter(0,0, c=colors, s=20)
dscatter =  ax.scatter(0,0, c="green", s=5)
wmscatter = ax.scatter(0,0, c=(0,0.6,0), s=100)
ax.set_xlim([-1.2, 1.2]); ax.set_ylim([-1.2, 1.2])
fig_loss = plt.figure()
ax_loss = fig_loss.add_subplot(111)
loss_plot, = ax_loss.plot(range(epochs),np.zeros(epochs))
ax_loss.set_ylim([0,5]) 


graph = tf.Graph()
with graph.as_default():

    ssom = SOM(input_num, output_num, batch_num, w_stddev=0.01, 
            min_modulation=1.0, reproduct_deviation=rep_deviation)

    ssom.generate_closed_graph()

    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        losses = np.zeros(epochs)
        # train

        target_idcs = np.arange(output_num)
        for epoch in range(epochs):
            
            if  epoch%output_num == 0: np.random.shuffle(target_idcs)

            curr_target_idx = (epoch%output_num)*np.ones(batch_num).astype(int)
            idx = target_idcs[curr_target_idx]
            curr_means = means[idx]/10.0 - 0.5
            curr_out = rotated[idx]/10.0 - 0.5

            curr_data = 0.02*np.random.randn(batch_num, 2) + curr_means
            
            norms, outs , loss_ = ssom.train_step(
                    curr_data, 
                    som_learning_rate*np.exp(-epoch/(epochs*0.075)),
                    centroids[idx[0]].reshape(1, output_num),
                    session)
            
            print "epoch:%4d       loss:% 10.3f" % (epoch, loss_)
                       
            W_ = ssom.W.eval()
            
            wscatter.set_offsets(W_.T)
            wmscatter.set_offsets(curr_means[0])
            wiscatter.set_offsets(W_[:,idx[0]])
            dscatter.set_offsets(curr_data)
            
            W_ = W_.reshape(2, output_side, output_side) 
            for i in range(output_side):
                wgrid[0][i].set_data(*W_[:,:,i])
            for i in range(output_side):
                wgrid[1][i].set_data(*W_[:,i,:])
            
            fig.canvas.draw()
            fig.savefig("frames/k%06d.png" % epoch)
                
            losses[epoch] = loss_.mean() 
            
            loss_plot.set_ydata(np.log(losses))     
            fig_loss.canvas.draw()
            fig_loss.savefig("loss.png")    
        

