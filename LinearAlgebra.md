# Engineering-Notes
2020 Engineering Notes


# Lily Das

<img src='https://avatars.githubusercontent.com/u/49919045?v=4' width=100>





__3.__
Part 1: What is the arithmetic intensity for matrix matrix multiplication
with sizes M, N and K (BLAS notation)? Part 2: Prove that arithmetic intensity for matrix matrix
multiplication is maximized when M = N = K (all matrices are square).

__Solution:__ While multiplying two matrices we have the following operations:

- N multiplications across K rows M times -> MNK MACs
- 2 Loads MK, KN and one store MN -> MK+KN+MN Mem Ops

<center>$Arithmetic Intensity = \frac{MNK}{(MK+KN+MN)}$</center>

<br>

<br>

__4.__
Consider a dense layer that transforms input feature vectors to output feature vectors of the
same length (No = Ni). Ignoring the bias and pointwise nonlinearity, what is the complexity
(MACs and memory) of this layer applied to inputs created from vectorized versions of the
following:

- MNIST: 1 x 28 x 28
- CIFAR: 3 x 32 x 32
- ImageNet: 3 x 224 x 224 (typical use)
- Quasi 1/4 HD: 3 x 512 x 1024
- Quasi HD: 3 x 1024 x 2048

__Solution:__
Complexity is $\frac{K^{2}}{2K+K^{2}}$ Because $K=N$


def solve_question_four(C,H,W):
    K = C*H*W
    return K**2 , (2*K + K**2)

print("Dataset:  (MACs, Memory)")
print("MNIST: ", solve_question_four(1,28,28),)
print("CIFAR: ", solve_question_four(3,32,32))
print("ImageNet: ", solve_question_four(3,224,224))
print("Quasi 1/ HD: ", solve_question_four(3,512,1024))
print("Quasi HD: ", solve_question_four(3,1024,2048))

Dataset:  (MACs, Memory)
MNIST:  (614656, 616224)
CIFAR:  (9437184, 9443328)
ImageNet:  (22658678784, 22658979840)
Quasi 1/ HD:  (2473901162496, 2473904308224)
Quasi HD:  (39582418599936, 39582431182848)

__5.__ 
In practice, why can’t you flatten a quasi HD input image (3 x 1024 x 2048) to a 6291456 x 1
vector and use densely connected layers to transform from data to weak features to strong features to classes?

__Solution:__
Way to many Parameters. This would hinder training due to the number of operations that need to be performed as well as the sheer memory requirements of having to store not only the parameters, but every intermediate product and gradient.

__6.__
Say I have trained a dense layer for an input of size 1024 x 1. Can this dense layer be applied
to an input of size 2048 x 1? What about 512 x 1?

__Solution:__ Fixed size input, the input image resolution would have to be the same as the training data.

__8.__
Consider a CNN style 2D convolution layer with filter size No x Ni x Fr x Fc. How many MACs
are required to compute each output point?

__Solution:__ So based on the wording of this question, I am assuming that the operation will be one step in the convolution of a single filter.

MACs: $F_r*F_c*N_i$



__9.__
How does CNN style 2D convolution complexity (MACs and memory) scale as a function of
- Product of the image rows and cols (Lr*Lc)?
- Product of the filter rows and cols (Fr*Fc), assume Ni and No are fixed?
- Product of the number of input and output feature maps (Ni*No)?

__Solution:__



__10.__
Consider a CNN style 2D convolution layer with filter size No x Ni x Fr x Fc. How many 0s do I
need to pad the input feature map with such that the output feature map is the same size as the
input feature map (before 0 padding)? What is the size of the border of 0s for Fr = Fc = 1? What
is the size of the border of 0s for Fr = Fc = 3? What is the size of the border of 0s for Fr = Fc = 5?

__SOlution:__
Row Pad: $F_r-1$ Col Pad: $F_c-1$

No border required

1 pixels thick on both rows and columns.

2 pixels thick on rows and columns


__11.__
Consider a CNN style 2D convolution layer with No x Ni x Fr x Fc filter, Ni x Lr x Lc input (Lr and
Lc both even) and Pr = Fr – 1 and Pc = Fc – 1 zero padding. What is the size of the output feature
map with striding Sr = Sc = 1 (no striding)? What is the size of the output feature map with
striding Sr = Sc = 2? How does this change the shape of the equivalent lowered matrix equation?

__SOlution:__

No Striding: $N_o*L_r*L_c $

Striding=2: $N_o*\frac{L_r*L_c}{2}$

The filter tensor doesn't change. For the input map something like: Keep the first of every three groups of columns of length W. Within the kept group of tensors, keep the first column of every three.



__12.__
Say I have trained a CNN style 2D convolution layer for an input of size 3 x 1024 x 2048. Can
this CNN style 2D convolution layer be applied to an input of size 3 x 512 x 1024? What about 3
x 512 x 512?

__Solution:__
Yes, the convolutional layer is compatable with different sized inputs (except the channel dimension).


__RNN layers__


__13.__
In a standard RNN, if the state update matrix is constrained to a diagonal, what does this do
for the mixing of the previous state with new inputs?


__Solution:__
This would be one big linear layer, since there is no mixing of information across time.

__Attention layers__




__14.__
Consider single headed self attention where input XT is a M x K matrix composed of M input
vectors with K features per vector, Ai
T is a M x M attention matrix where each element is non
negative and each row sums to 1 (think each row is a pmf), Wv,i is a K x L weight matrix and
output Yi
T is a M x L matrix composed of M output vectors with L features per vector
Yi
T = Ai
T XT Wv,i
Can Ai
T be computed once and then used for all inputs? What does it mean for the output if Ai
T
is an identity matrix? What does it mean for the output if Wv,i is an identity matrix?

__Solution__
No the formula for A depends on X.

Then the output of the Attention* input multiplication is identical to the input matrix -> Linear layer without a bias

We will have mixing across inputs but no mixing. Not really sure how this would work.


#### Average pooling layers
__15.__
The size of the input to a global average pooling layer is 1024 x 16 x 32. What is the size of
the output? What is the complexity (operations) of the layer?

__Solution:__

1024x1

$16*32$ operations for each of the 1024 channels.


[I need to execute the MNIST Code on AWS]



