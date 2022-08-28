# Engineering-Notes
2020 Engineering Notes
# Optimization

## Loss Functions and Optimization (Lecture 3)

### Sources
- [linear](http://cs231n.github.io/linear-classify/)
- [opt-1](http://cs231n.github.io/optimization-1/)
- [lecture video](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4&t=0s)
- [lecture3 Slides](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture03.pdf)

__Linear classification intuition__ 

Each row in the matrix of weights represents a template for each class. Each weight represents the amount that each pixel contributes to the class prediction. Can also be seen as a high dimensional linear decision boundary seperating the vectors where values are pixel intensities.

$$f(x,W,b)=Wx+b$$

In the case of multiclass classification, W represents a matrix with a row for each potential target class. By multiplying this tensor with the input data x, the model evaluates the score for each class concurrently.

Because of their limitations, linear classifications can only pick up on very simple features (color and location).

__Loss Functions__ Maps a prediction and a true output label to a scalar that increases as the model performs worse. __Soft constraints__ on the model can be applied by adding regularization terms to the loss function. (L1, L2, ...) 

### __SVM Classifiers__

$$Hinge Loss_i = \sum max(0,s_j-s_{y_i}+\Delta)+\lambda R(W)$$

Here $y_i$ represents the index of the correct class. $s_j$ represents the class scores determined by the linear classifier. The linear model predicts the highest class, so the above function penalizes any prediction that scores higher than the true class label. Any negative score receives a loss of zero.

def multiclass_svm_loss(x,y, W,delta=1, squared=False, l2_penalty=0.0):
    """
    x: a tensor shape (1,feat_dims)
    W: a tensor shape (feat_dims, num_classes)
    y: a vector with the index of correct class 
    delta: goal margin between prediction class and other classes
    squared: whether to use squared hinge
    l2_penalty: regularization strength for L2 penalty
    """
   
    scores = tf.linalg.matmul(x, W)
    margins = tf.maximum(0, scores-scores[y]+delta)
    margins[y]=0
    
    if squared:
        margins = tf.math.square(margins)
        
    #Add L2 penalty    
    margins+=l2_penalty*tf.reduce_sum(tf.math.square(W),axis=1, keep_dims=True)
    
    loss_i = tf.reduce_sum(margins)
    return loss_i
    
    ### __SoftMax Classifier__
Alternative to an SVM classifier. The output of a softmax classifier is a probability distribution over the potential classes. With a softmax classifier, we interpret $f(x)=W\cdot x$ as __unnormalized log probabilities__ for each class.

__Cross-Entropy Loss__

$$Cross Entropy Loss = -log(\frac{e^{f_{y_i}}}{\sum_je^{f_j}})$$


__Cross Entropy__
$$Cross Entropy = -\sum p(x)log(q(x))$$

__SoftMax Function__

$$f(score) - \frac{e^{score_{j}}}{\sum_ke^{score_k}}$$

Softmax classifiers minimizes the crossentropy between the true distribution and the predicted distribution.

*Moreover, since the cross-entropy can be written in terms of entropy and the Kullback-Leibler divergence as $H(p,q)=H(p)+D_{KL}(p||q)$, and the entropy of the delta function p is zero, this is also equivalent to minimizing the KL divergence between the two distributions (a measure of distance). In other words, the cross-entropy objective wants the predicted distribution to have all of its mass on the correct answer.*


def softmax(logits, axis=-1):
    """Naive Implementation"""
    exps = tf.math.exp(logits)
    return exps/tf.reduce_sum(exps, axis)

#tf.nn.softmax()


def crossentropy_loss(x, y, W, stable=True):
    """
    x: a tensor shape (1,feat_dims)
    W: a tensor shape (feat_dims, num_classes)
    y: a vector with the index of correct class
    stable: whether to scale scores for numeric stability
    """
    score = tf.linalg.matmul(x,W)
    if stable: #scale by max value
        score-=tf.reduce_max(score)
    probs = softmax(score)
    return -tf.math.log(scores)[y]

#tf.keras.losses.sparse_categorical_crossentropy()


### Regularization (weight decay)

Generally takes the form of
$$L(W)=\frac{1}{N}\sum_i L_i(f(x_i,W),y_i)+\lambda R(W)$$

def reg(loss_func, x, y, W, l1=0.0, l2=0.0):
    """
    Computes Elastic Net regularization for an arbitrary loss function of
    form L(x,y,W)
    """
    return loss_func(x,y,W) + l2*tf.reduce_sum(tf.math.square(W)) + l1*tf.reduce_sum(tf.math.abs(W))
    
    ### __Gradient Computation__ 

One can compute the gradient __numerically__ by evaluating the function with small perterbations along each dimension, or __analytically__ by deriving the gradient function with calculus. Numerical gradient computations can often be used as a sanity check of analytical methods.


def mult_n_add(x, w):
    return tf.linalg.matmul(x, w) + 4

def numerical_gradient(func, x, w):
    """
    func: function to approximate gradient
    x: a tensor shape (1,feat_dims)
    """
    fx = func(x,w)
    h=1e-3
    #create a diagonal to add delta
    diff_matrix = (tf.eye(x.shape[-1]) * h) + w
    #wanted to use tf.map_fn here, but couldnt with the x input
    #unstack -> recover dim -> apply function subtract by unperterbed value
    grads = [func(x, tf.expand_dims(g,-1)) - fx for g in tf.unstack(diff_matrix, axis=-1)]
    #stack and throw out extra dim
    return tf.squeeze(tf.stack(grads), axis=-1)/h
    
#tf.test.compute_gradient()

x = tf.random.normal((1,3))
w = tf.random.normal((3,1))

with tf.GradientTape() as g:
    g.watch(w)
    y = mult_n_add(x,w)
dy_dw=g.gradient(y,w)
    
print("Auto gradient: ",dy_dw)

Auto gradient:  tf.Tensor(
[[-0.9707055]
 [ 0.9031635]
 [-1.1831928]], shape=(3, 1), dtype=float32)
 
 numerical_gradient(mult_n_add, x, w)
 
 <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
array([[-0.9708404],
       [ 0.9031295],
       [-1.1835098]], dtype=float32)>
       
       
