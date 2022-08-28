# Engineering-Notes
2020 Engineering Notes

## Introduction to Neural Networks 

### Sources
- [optimization-2](http://cs231n.github.io/optimization-2/)
- [deriv-paper](http://cs231n.stanford.edu/handouts/derivatives.pdf)
- [lecun paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
- [lecture4slides](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture04.pdf)
- [lecture video](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4)
- [auto-diff-survey](https://arxiv.org/pdf/1502.05767.pdf)

### Computational Graphs ([video](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4))
__Nodes__ are the computations and __edges__ are tensors. Backpropagation uses the chain rule to express the gradient wrt to every variable in the computational graph.

Start at the computed loss at the end of the computational graph. $\frac{\delta Loss}{\delta z}$. Then use the chain rule
$$\frac{\delta f}{\delta y} = \frac{\delta f}{\delta q}\frac{\delta q}{\delta y}$$

For an arbitrary function $q(x,w)$ we can insert the function into the computational graph and apply backprop as long as we have the __local gradient__ $\frac{\delta q}{\delta x}$ and $\frac{\delta q}{\delta w}$

You can make nodes in the graph have any granularity. They can be distinct additions, multiplications, or the grouping of multiple operations into a single node. ex make a sigmoid node. Grouping operations can make you comp graph smaller.

#SIGMOID AS A SINGLE NODE
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

z=tf.constant([0.2])
with tf.GradientTape() as tape:
    tape.watch(z)
    sig = sigmoid(z)
dsig_dz = tape.gradient(sig,z)

#see below for a plot of sigmoid gradient
print(dsig_dz)


tf.Tensor([0.24751654], shape=(1,), dtype=float32)


#ADD GATE PASSES BACK GRADIENT TO BOTH BRANCHES
def add_gate(x):
    return tf.reduce_sum(x)

z=tf.constant([0.2, 3.])
with tf.GradientTape() as tape:
    tape.watch(z)
    m = add_gate(z)
dm_dz = tape.gradient(m,z)

print(dm_dz)

tf.Tensor([1. 1.], shape=(2,), dtype=float32)

#MAX GATE ROUTES GRADIENT TO MAX ELEMENT   
def max_gate(x):
    return tf.reduce_max(x)

z=tf.constant([.2, 3.])
with tf.GradientTape() as tape:
    tape.watch(z)
    m = max_gate(z)
dm_dz = tape.gradient(m,z)

print(dm_dz)

tf.Tensor([0. 1.], shape=(2,), dtype=float32)


#MULTIPLICATION GATE - LOCAL GRADIENT IS THE VALUE OF THE OTHER BRANCH
def mult_gate(x):
    return tf.reduce_prod(x)

z=tf.constant([0.2, 3.])
with tf.GradientTape() as tape:
    tape.watch(z)
    m = mult_gate(z)
dm_dz = tape.gradient(m,z)

print(dm_dz)

tf.Tensor([3.  0.2], shape=(2,), dtype=float32)

__At branches__ in the computational graph, the gradients sum together during back propagation.

$$Vec(1,4000)\rightarrow f(x)=max(0,x) \rightarrow Vec(1,4000)$$

$$Jacobian \in \mathbb{R} (4000,4000)$$

__Jacobian__ each row is partial derivative of each dim of output wrt to each dim of input. In practice dont need to compute a huge Jacobian $\rightarrow$ Jacobian is going to be a diagonal matrix for elementwise functions. 

__The gradient wrt a vector is always going to be the same size of the original vector__. Each element in the gradient shows how much each corresponding weight/input effects the ouput of the computational graph. 


#### Implementing in Code
During forward pass __compute nodes in topologically sorted order__ so every input is ready when it is needed. __Cache the values of the forward pass for use in backwards pass.__

Implement the graph with node classes with `.forward()` and `.backward()` API.

class sigmoid_node():
    """Sigmoid operation"""
    def __init__(self, axis=-1):
        self.z=None
    def forward(self, x):
        self.z = 1 / (1 + tf.exp(-x))
        return self.z
    def backward(self, dz):
        return dz*(1-self.z)*self.z

class add_node():
    """Elementwise addition"""
    def __init__(self,):
    def forward(self,x,y):
        return tf.add(x,y)
    def backward(self,dz):
        return [dz,dz]

class max_node():
    def __init__(self, axis=-1):
        self.max_val=None
        self.mat_shape=None
        self.axis=axis
    def forward(self,x):
        self.max_val=tf.argmax(x, axis=self.axis)
        self.mat_shape=x.shape
        return tf.reduce_max(x, axis=self.axis)
    
    def backward(self,dz):
        g = tf.zeros(self.mat_shape)
        g[self.max_val] = dz #this wont work, but im not sure how to set indexed values
        return g
    
class mult_node():
    def __init__(self,):
        self.x=None
        self.y=None
    def forward(self,x, y):
        self.x=x
        self.y=y
        return x*y
    def backward(self,dz):
        dy=dz*self.x
        dx=dz*self.y
        return [dx,dy]

class mat_mul_node():
    def __init__(self,):
        self.x=None
        self.y=None
    def forward(self,x, y):
        self.x=x
        self.y=y
        z = tf.matmul(x,y)
        return z
    def backward(self,dz):
        dy=tf.matmul(dz, self.x, transpose_b=True)
        dx=tf.matmul(dz, self.y, transpose_b=True)
        return [dx,dy]
        
        
        class computational_graph():
    def __init__(self):
        #CONSTRUCT GRAPH HERE
        return None
    def call(self, inputs):
        #ITERATE THROUGH GRAPH WITH .forward() METHOD
        return None
    def backward(self):
        #ITERATE THROUGH GRAPH WITH .backward() METHOD
        return None
        
        __Stacking nonlinear functions__ allows stacking simpler functions together to compute complex nonlinear functions. $W_1$ is like a template for each predicted class. $W_1$ is a weighting of these templates that allows combinations of features. 
$$f=W_2max(0,W_1\cdot x)$$


# Neural Networks

## Convolutional Neural Networks (Lecture 5)

### Sources
- [lecture-slides](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture05.pdf)
- [cnn-notes](http://cs231n.github.io/convolutional-networks/)
- [lecture-video](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5)

Learned this too many times. Moving forward.

### Brief CNN Notes ([cnn-notes](http://cs231n.github.io/convolutional-networks/))
*__Local Connectivity__. When dealing with high-dimensional inputs such as images, as we saw above it is impractical to connect neurons to all neurons in the previous volume. Instead, we will connect each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the __receptive field__ of the neuron*

*The connections are local in space (along width and height), but always full along the entire depth of the input volume.*

How are kernels/filters applied to the input data? Three parameters determine the size of the activation for a given input.

*First, the depth of the output volume is a hyperparameter: it corresponds to the __number of filters__ we would like to use, each learning to look for something different in the input.*

*Second, we must specify the __stride__ with which we slide the filter.*

*As we will soon see, sometimes it will be convenient to pad the input volume with zeros around the border. The size of this __zero-padding__ is a hyperparameter.*

__Parameter Sharing__ Allows conv nets to be more efficient than standard FC networks. Instead of connnecting a neuron to every pixel, share features that can be convolved across the image. Assumes that fe

__1x1 convolutions__ Linear combination layer across the dimensions of the input. Often used to expand or shrink the depth dimension.

__Pooling Layers__ progressively reduces the spatial volume of feature maps and reduces the amount of parameters (and therefore computation)

*Getting rid of pooling. Many people dislike the pooling operation and think that we can get away without it. For example, Striving for Simplicity: The All Convolutional Net proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers.*

### Network in Network ([paper](https://arxiv.org/pdf/1312.4400v3.pdf)) 2014

__Network in Network__ is the stacking of multiple mlpconv layers. This involves a spatial convolution followed by a series of fully connected layers with nonlinearities that share features across the spatial regions.

Spatial conv $\rightarrow$ __1x1 convolutions__

*The cross channel parametric pooling layer is also equivalent to a convolution layer with 1x1 convolution kernel. This interpretation makes it straightforawrd to understand the structure of NIN.*

*This cascaded cross channel parameteric pooling structure allows complex and learnable
interactions of cross channel information.*

*In this paper, we propose another strategy called __global average pooling__ to replace the traditional
fully connected layers in CNN. The idea is to generate one feature map for each corresponding
category of the classification task in the last mlpconv layer. Instead of adding fully connected layers
on top of the feature maps, we take the average of each feature map, and the resulting vector is fed
directly into the softmax layer. One advantage of global average pooling over the fully connected
layers is that it is more native to the convolution structure by enforcing correspondences between
feature maps and categories. Thus the feature maps can be easily interpreted as categories confidence
maps.*

class NetworkinNetwork(tf.keras.layers.Layer):
    def __init__(self, unit_list, num_mlp_layers=3):
        super(mlpConv, self).__init__()
        self.spatial_conv = tf.keras.layers.Conv2D(unit_list[0], 3, padding='same')
        self.mlp_layers = []
        self.num_mlp_layers = num_mlp_layers
        for i in range(num_mlp_layers):
            self.mlp_layers.append(tf.keras.layers.Conv2D(unit_list[i],1, padding='same', activation='relu'))
    
    def call(self, inputs):
        x = self.spatial_conv(inputs) 
        for i in self.num_mlp_layers:
            x = self.mlp_layers[i](x)
        return x
        
 ### All Conv Net ([paper](https://arxiv.org/pdf/1412.6806.pdf) 2015)

*We find that max-pooling can simply be replaced by a convolutional layer with increased stride without loss in accuracy on several image recognition benchmarks. Following this finding – and building on other recent work for finding simple network structures – we propose a new architecture that consists solely of convolutional layers and yields competitive or state of the art performance on several object recognition datasets*

THeres some other interesting stuff here about the use of 1x1 convolutions and deconvoltions for kernel visualization.
       
