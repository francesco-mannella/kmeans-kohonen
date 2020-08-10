### imports
import matplotlib.pyplot as plt
import numpy as np
import shutil, time
import tensorflow as tf
from modulated_kohonen import SOM

#-------------------------------------------------------------------------------

class TimeOverflow(Exception): pass

#-------------------------------------------------------------------------------
# only current needed GPU memory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#-------------------------------------------------------------------------------

np.set_printoptions(suppress=True, precision=3, linewidth=999)

#-------------------------------------------------------------------------------

class DeepSOM(object):

    def __init__(self, layers_len,  output_functions, som_output_channels, batch_num, 
            weight_scale=0.02, som_w_mean=0.0, som_w_stddev=0.03, som_neighborhood=0.7, 
            scope="deep_som"):
        """
        :param layers_len: list(int), number of units per layer
        :param output_functions: list(callable), list of activation functions for each layer  
        :param batch_num: number of patterns presented in a single batch
        :weight_scale:  the initial scaling of mlp weights
        :param som_output_channels: length of the vector of output units of the SOM
        :param som_w_stdder: standard deviation og initial weight distribution of the som
        :param som_neighborhood: standard deviation of radial bases from som winners
        """
        self.scope = scope
        self.layers_len = layers_len
        self.output_functions = output_functions

        # mlp network   
        self.weights = []
        self.biases = []
        self.layers = []
    
        with tf.variable_scope(scope, auxiliary_name_scope=False):
            with tf.variable_scope("mlp", auxiliary_name_scope=False):
                for l0,_ in enumerate(output_functions):
                    with tf.variable_scope("layer-{:03d}".format(l0), auxiliary_name_scope=False):
                        
                        # weight and bias initializers
                        weight_shape = (self.layers_len[l0], self.layers_len[l0+1])
                        bias_shape = (self.layers_len[l0+1], )

                        w_initial = tf.truncated_normal(weight_shape, stddev=weight_scale)
                        b_initial = tf.constant(0.0, shape = bias_shape)
                        weight = tf.get_variable(name="w", dtype=tf.float32,
                            initializer=w_initial) 
                        bias = tf.get_variable(name="b",dtype=tf.float32, 
                                initializer=b_initial)
                        
                        # store variables
                        self.weights.append(weight)
                        self.biases.append(bias)

            # SOM building graph
            self.som = SOM(
                    input_channels=self.layers_len[-1],
                    output_channels=som_output_channels,
                    batch_num=batch_num, 
                    scope="som")

    def spreading_graph(self, inp):
        self.layers = []
        self.layers.append(inp)
        
        with tf.variable_scope(self.scope, reuse=False,  auxiliary_name_scope=False):
            with tf.name_scope("mlp_spreading"):
                with tf.variable_scope("mlp", reuse=False,  auxiliary_name_scope=False):
                    for l0, outfun in enumerate(self.output_functions):
                        layer_name = "layer-{:03d}".format(l0)
                        with tf.name_scope(layer_name):
                            with tf.variable_scope(layer_name, reuse=False, auxiliary_name_scope=False):
                                with tf.name_scope("activation"):
                                    ofun = self.output_functions[l0]
                                    # layer spreading graph
                                    weight = self.weights[l0]
                                    bias = self.biases[l0]
                                    input_layer = self.layers[l0]   
                                    output_layer = ofun(tf.matmul(input_layer, weight) + bias)     
                                    # store layer
                                    self.layers.append(output_layer)

            self.norms, self.rk_bases, self.out_bases = self.som.spreading_graph(self.layers[-1])
    
    def backpropagate_graph(self, output_points):
        with tf.variable_scope(self.scope, reuse=False,  auxiliary_name_scope=False):    
            generated_som_inputs, _ = dsom.som.backpropagate_graph(output_points=output_points)
            output_layer = generated_som_inputs
            for l0, outfun in list(enumerate(self.output_functions))[::-1]:
                weight = self.weights[l0]
                bias = self.biases[l0]
                print(l0, output_layer.get_shape(), weight.get_shape(), bias.get_shape())
                input_layer = tf.matmul(output_layer - bias, tf.transpose(weight))     
                output_layer = input_layer
        
        return input_layer


    def training_graph(self, modulation=None, optimizer=tf.train.AdamOptimizer,
            learning_rate=0.001):    

       
        with tf.variable_scope(self.scope, reuse=False, auxiliary_name_scope=False):
            
            if modulation is None: 
                modulation = self.som.rk_bases
                
            self.loss = self.som.compute_loss(self.som.norms, modulation)
            
            # gradient descent
            self.train = optimizer(
                    learning_rate).minimize(
                    self.loss, var_list=self.weights + [self.som.W], 
                    name="Optimizer")

        return self.loss, self.train

#------------------------------------------------------------------------------
# plotting 

class Plots:
    def __init__(self):
        self.fig1 = plt.figure(figsize=(6, 3))
        self.ax1 = self.fig1.add_subplot(111)
        self.plt1, = self.ax1.plot(np.zeros(10), np.zeros(10))    

        self.fig2 = plt.figure(figsize=(5, 5))
        num_sampling = 100
        rows_sampling = num_sampling//10
        cols_sampling = 10
        self.plt2 = []
        for k in range(num_sampling):
            ax2 = self.fig2.add_subplot(rows_sampling, cols_sampling, k+1)
            ax2.set_axis_off()
            self.plt2.append(ax2.imshow(np.zeros([28,28]), vmin=0, vmax=1))

    def __call__(self,losses, out_sampling):
        # plot loss
        
        try:
            self.plt1.set_data(np.arange(len(losses)), np.hstack(losses))
            self.ax1.set_xlim([0, len(losses)])
            self.ax1.set_ylim([0, np.max(losses)])
            self.fig1.canvas.draw() 
        except:
            pass

        num_sampling = 100
        for k in range(num_sampling):
            self.plt2[k].set_array(out_sampling[k].reshape(28,28))
         
        plt.pause(0.01)



if __name__ == "__main__":
    plt.ion()

    #-------------------------------------------------------------------------------
    # load MNIST dataset

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    shutil.rmtree("MNIST-data")

    #------------------------------------------------------------------------------
    # parameters

    data_num = len(train_data)
    batch_num = 2500
    side = 28
    input_channels = side*side
    output_channels = 5*5
    epochs = 600
    initial_learning_rate = 0.05

    sim_time = 800000
    start_time = time.time()

    graph = tf.Graph()
    with graph.as_default():
        
        X = tf.placeholder(tf.float32, (None, input_channels))
        learning_rate = tf.placeholder(tf.float32, ())
        dsom = DeepSOM([input_channels, 5, 10], [tf.nn.relu, tf.tanh],
                output_channels, batch_num,  weight_scale=0.0002,  scope="mps")
        
        dsom.spreading_graph(X)
        dsom.training_graph(learning_rate=learning_rate)
        output_points = tf.placeholder(tf.float32, (None, 2))
        generated_inputs = dsom.backpropagate_graph(output_points)
       
        with tf.Session(config=config) as session:
            tf.global_variables_initializer().run()
            file_writer = tf.summary.FileWriter('./tb', session.graph)
            
            losses = []

            try:

                # train
    
                plots = Plots()

                for epoch in range(epochs):
                    
                    np.random.shuffle(train_data)
                    
                    # decaying deviation 
                    curr_deviation = output_channels*np.exp(-epoch/float(epochs/6.0))
                    print(curr_deviation)
                    # decaying learning rate 
                    curr_learning_rate = initial_learning_rate*np.exp(-epoch/float(epochs))

                    # run a batch of train steps 
                    elosses = []

                    for batch in range(data_num//batch_num):
                        curr_time = time.time() - start_time
                        if curr_time >= sim_time:
                            raise TimeOverflow("No time left!")
                        
                        print(("epoch:%4d       batch:%4d" % (epoch, batch)))
                        
                        curr_batch =  train_data[batch * batch_num : (batch + 1) * batch_num ,:]
                        
                        norms_, y_ ,loss_, _ = session.run([dsom.norms, dsom.out_bases, 
                                dsom.loss, dsom.train], feed_dict={ X: curr_batch, 
                                    learning_rate: curr_learning_rate})

                        elosses.append(loss_)

                    losses.append(np.mean(elosses))


                    if epoch % 2 == 0:
                        # test
                        
                        g_means = np.random.uniform(0, 10, [100,2])
                        sampling = session.run(generated_inputs, 
                                feed_dict={output_points: g_means})
                        sampling = sampling.reshape(100,  side, side)
                        plots(losses, sampling)


            except TimeOverflow:
                
                print("Plotting partial results...")
   
