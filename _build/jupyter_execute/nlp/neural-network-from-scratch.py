# Neural Network From Scratch

This notebook provides an intuitive understanding of the mechanism of the neural network, or deep learning.

Important steps in neural network:

- forward propagation
    - matrix multiplication
    - weights, biases, and activation functions
- back propagation
    - derivatives and partial derivatives
    - chain rules
- gradient descent 
    - Batch
    - Mini-batch
    - Stochastic gradient descent

---

![](../images/neural-network-sample.png)

---

import numpy as np
import matplotlib.pylab as plt

## Linear Algebra and Matrix

- 2D matrix

$$
\begin{pmatrix}
1&2 \\
3&4 \\
5&6 \\
\end{pmatrix}
$$

- Matrix Multiplication


$$\begin{pmatrix}
1&2 \\
3&4 \\
\end{pmatrix}
\begin{pmatrix}
5&6 \\
7&8
\end{pmatrix} =
\begin{pmatrix}
19&22 \\
43&50
\end{pmatrix}
$$

## Activation Functions

In neural network, the activation function of a node determines whether the node would activate the output given the input values. Different types of activation functions may determine the cut-offs for output activation in different ways.

- sigmoid function

$$ h(x) = \frac{1}{1 + \exp(-x)}$$

- step function

$$ h(x)= \left\{ 
     \begin{array}\\
     0 & (x \leq 0) \\
     1 & (x > 0)
     \end{array}
\right.
$$

- ReLU function

$$ h(x)= \left\{ 
     \begin{array}\\
     x & (x > 0) \\
     0 & (x \leq 0)
     \end{array}
\right.
$$

- Softmax function


$$
y_k = \frac{\exp(a_k)}{\sum_{i = 1}^{n} {a_i}}
$$

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)


# def softmax(x):
#     exp_x = np.exp(x)
#     sum_exp_x = np.sum(exp_x)
#     y = exp_x/sum_exp_x
#     return y

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c) # avoid overflow issues
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x
    return y

# step function
x = np.arange(-5.0, 5.0, 0.1)
plt.plot(x, step_function(x))
plt.ylim(-0.1, 1.1)
plt.show()

## sigmoid function
plt.plot(x, sigmoid(x))
plt.ylim(-0.1, 1.1)
plt.show()

# ReLU
plt.plot(x, relu(x))
plt.ylim(-1, 6)
plt.show()

## Forward Propagation

![](../images/neural-network-sample2.png)

- Neural network is a model with weights for data/value transformation.
- The input data values will be transformed according to the weights of the neural network.
- Given a two-layer network, with two input values $x1$ and $x2$, to get the values of the three outputs in the second layer, $a_1^{(1)}$, $a_2^{(1)}$, $a_3^{(1)}$, we compute the dot product of the *X* and *W*.
    - *X* refers to the input vector/matrix
    - *W* refers to the network weights, which is a 2 x 3 matrix in the current example
    - The weights are represented as the links in-between the first and second layers
    - These weights can be mathematically represesnted as a 2 x 3 Matrix *W*
- Taking the dot product of the input values *X* and the weight matrix *W* is referred to as the **forward propagation** of the network.
- Forward propagation gives us the values of the nodes in the second layer

X = np.array([1,2])
X.shape

W = np.array([[1,3,5],[2,4,6]])
W.shape

Y = np.dot(X,W)
print(Y)

## Weights, Biases, and Activation Functions

- The output of a node in the network is computed as the sum of the weighted inputs and the bias. Take $a^{(1)}_1 $ for example:

$$ a^{(1)}_1 = w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + b_1$$

- Then the output values go through the activation function and this result would indicate the final output of the node.

$$ z^{(1)}_1= h(a^{(1)}_1) $$

- Not all the nodes need to have an activation function.

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1,0.2,0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)


Z1 = sigmoid(A1)
print(Z1)

Z2 = softmax(A1)
print(Z2)

## Learning and Training

- Forward propagation shows how the network takes the input values and produce the output values based on the network parameters (i.e., weights).
- The network needs to learn the weights that best produce the output values according to some loss function.

## Loss Function on One Sample

- Mean Square Error 

$$E = \frac{1}{2}\sum(y_k - t_k)^2$$

- Cross Entropy Error

$$E= -\sum_{k}t_k\log(y_k)$$

def mean_square_error(y, t):
    return(0.5 * np.sum((y-t)**2))

def cross_entropy_error(y, t):
    delta = 1e-7 # avoid log(0)
    return -np.sum(t * np.log(y + delta))

## mean square error
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # predicted values
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # true label
print(mean_square_error(np.array(y),  np.array(t)))
print(cross_entropy_error(np.array(y), np.array(t)))

## Loss Function on Batch Samples

- If the training goes based on a small batch size *N*, we can compute the average loss of the batch sample:

$$ E = - \frac{1}{N}\sum_n\sum_k t_{nk}\log y_{nk}$$

- We can revise the `cross_entropy_error()` function to work with outputs from a min-batch sample.

# adjust the function to for batch sample outputs
def cross_entropy_error(y, t):
    if y.ndim==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7))/batch_size 

- When the labels uses one-hot encoding, the function can be simplified as follows:

def cross_entropy_error(y, t):
    if y.ndim==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # because for one-hot labels
    # cross-entropy sums only the values of the true labels `1`
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))/batch_size

## Gradient Descent



## References

- [Deep Learning From Scratch](https://www.books.com.tw/products/0010761759)
- [史上最完整機器學習自學攻略！我不相信有人看完這份不會把它加進我的最愛](https://buzzorange.com/techorange/2017/08/21/the-best-ai-lesson/)