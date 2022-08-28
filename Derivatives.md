# Engineering-Notes
2020 Engineering Notes

### Derivatives ([Paper](http://cs231n.stanford.edu/handouts/derivatives.pdf))

__Gradient__ the derivative of a vector to scalar function. The gradient is a vector of the same size as the input vector.The gradient is a vector of partial derivatives.


$$f:\mathbb{R}^N\rightarrow \mathbb{R}$$


$$Gradient=(\frac{\delta y}{\delta x_1},...,\frac{\delta y}{\delta x_N})$$

__Jacobian__ the derivative of a vector to vector function. The jacobian is a matrix the shape of each of the input vectors. The Jacobian tells the relation between each element in x to each element in y. Each row in the Jacobian is a gradient to a single scalar element of y.

$$f:\mathbb{R}^N\rightarrow \mathbb{R}^M$$

<img src='imgs/jacobian.PNG' width=250>

__Generalized Jacobian__ The dimensions of the Jacobian will be the cross product of the dimensions of the input and output.

$$f:\mathbb{R}^{N_1x...xN_D}\rightarrow \mathbb{R}^{M_1x...xM_B}$$

$$Jacobian Dim: (M_1x...xM_B)x(N_1x...xN_D)$$


*Note that we have separated the dimensions of ∂y/∂x into two groups: the
first group matches the dimensions of y and the second group matches the
dimensions of x. With this grouping, we can think of the generalized Jacobian
as generalization of a matrix, where each  “row” has the same shape as y and
each “column” has the same shape as x.*

### Automatic Differentiation ([paper](https://arxiv.org/pdf/1502.05767.pdf))

* "All numerical computations are ultimately compositions of a finite set of elementary operations for which derivatives are known (Verma, 2000; Griewank and Walther, 2008), and combining the derivatives of the constituent operations through the chain rule gives the derivative of the overall composition.Usually these elementary operations include the binary arithmetic operations, the unary sign switch, and transcendental functions such as the exponential, the logarithm, and the trigonometric functions."*

AD can differnetiate closed-form expressions as well as algorithmic control flow with branches and loops. Control flow can be differentiated because any code execution will result in single flow of numerical computation w/ particular values for inputs, intermediate values, and output variables. 

* "AD is blind with respect to any operation, including control flow statements, which do not directly alter numeric values."*

__Forward Mode__ Associate each intermediate variable $v_1$ with a local derivative 

$$\dot{v_1}=\frac{\delta v_1}{\delta x_1}$$

After local derivatives are stored, the final Jacobian can be computed in multiple passes with the stored

Less storage required since intermediate dependencies do not need to be stored. 

__Reverse Mode__ reverse accumulation corresponds to the general backpropagation algorithm.

*In reverse mode AD, derivatives are computed in the second phase of a two-phase process. In the first phase, the original function code is run forward, populating intermediate variables vi and recording the dependencies in the computational graph through a bookkeeping procedure. In the second phase, derivatives are calculated by propagating adjoints v¯i in reverse, from the outputs to the inputs.*

Reverse Mode AD is computationally more efficient than forward mode, but requires more storage. In most cases backprop algorithms are used to compute the gradient of a function of form

$$f:\mathbb{R}^N\rightarrow \mathbb{R}$$

For large N, reverse mode AD provides a highly efficient method of gradient computation.

<img src=imgs/reverse_mode.PNG width=400>


__Newton's Method__ make use of the gradient $\nabla f$ and the Hessian $H_f$ with updates of the form $$\Delta w=-\eta H_f^{-1}\nabla f$$

Faster convergence at the cost of more compute.

*An important direction for future work is to make use of nested AD techniques in machine learning, allowing differentiation to be nested arbitrarily deep with referential transparency (Siskind and Pearlmutter, 2008b; Pearlmutter and Siskind, 2008). Nested AD is highly relevant in hyperparameter optimization as it can effortlessly provide exact hypergradients, that is, derivatives of a training objective with respect to the hyperparameters of an optimization routine *

### Efficient Backpropagation [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) old


__A Few Practical Tricks__
- Stochastic Learning has better results
- Batch learning is faster (with GPU parallelization)
- Shuffle the training sets 
- Normalize the inputs
