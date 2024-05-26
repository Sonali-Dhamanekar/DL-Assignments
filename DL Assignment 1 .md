1. What is the function of a summation junction of a neuron? What is threshold activation function?

If the sum of the two EPSPs (E1 + E2) depolarizes the postsynaptic neuron sufficiently to reach the threshold potential, a postsynaptic action potential results. Summation thus allows subthreshold EPSPs to influence action potential production.

The activation function compares the input value to a threshold value. If the input value is greater than the threshold value, the neuron is activated. It's disabled if the input value is less than the threshold value, which means its output isn't sent on to the next or hidden layer.

2. What is a step function? What is the difference of step function with threshold function?

Step Function: Step Function is one of the simplest kind of activation functions. In this, we consider a threshold value and if the value of net input say y is greater than the threshold then the neuron is activated. Mathematically, Given below is the graphical representation of step function.

The activation function compares the input value to a threshold value. If the input value is greater than the threshold value, the neuron is activated. It's disabled if the input value is less than the threshold value, which means its output isn't sent on to the next or hidden layer.

3. Explain the McCulloch–Pitts model of neuron.

The McCulloch-Pitts neural model, which was the earliest ANN model, has only two types of inputs — Excitatory and Inhibitory. The excitatory inputs have weights of positive magnitude and the inhibitory weights have weights of negative magnitude. The inputs of the McCulloch-Pitts neuron could be either 0 or 1.

They are binary devices (Vi = [0,1]) Each neuron has a fixed threshold, theta. The neuron receives inputs from excitatory synapses, all having identical weights. (However it my receive multiple inputs from the same source, so the excitatory weights are effectively positive integers.)

4. Explain the ADALINE network model.

ADALINE (Adaptive Linear Neuron) is an artificial neural network model proposed by Bernard Widrow and Ted Hoff in 1960. It is similar to the perceptron, but instead of a step activation function, it uses a linear activation function.

ADALINE is a supervised learning model used to perform binary classification and linear regression. The neural network consists of an input layer, an output layer and a feedback layer that adjusts the weights of the input layer according to the output obtained.

The objective of ADALINE is to minimise the mean square error (MSE) between the desired output and the actual output of the network. It does this by using the gradient descent algorithm to adjust the input layer weights.

ADALINE is a linear model, which means that it can only learn linear relationships between inputs and outputs. However, it can be used as a basic unit in more complex neural network models, such as multilayer neural networks.

5. What is the constraint of a simple perceptron? Why it may fail with a real-world data set?

A perceptron model has limitations as follows:
The output of a perceptron can only be a binary number (0 or 1) due to the hard limit transfer function. Perceptron can only be used to classify the linearly separable sets of input vectors. If input vectors are non-linear, it is not easy to classify them properly.

Perceptrons only represent linearly separable problems. They fail to converge if the training examples are not linearly separable. This brings into picture the delta rule.

6. What is linearly inseparable problem? What is the role of the hidden layer?

You cannot draw a straight line into the left image, so that all the X are on one side, and all the O are on the other. That is why it is called "not linearly separable" == there exist no linear manifold separating the two classes.

Hidden layers are the intermediary stages between input and output in a neural network. They are responsible for learning the intricate structures in data and making neural networks a powerful tool for a wide range of applications, from image and speech recognition to natural language processing and beyond.


7. Explain XOR problem in case of a simple perceptron.

As we can see from the truth table, the XOR gate produces a true output only when the inputs are different. This non-linear relationship between the inputs and the output poses a challenge for single-layer perceptrons, which can only learn linearly separable patterns.

A "single-layer" perceptron can't implement XOR. The reason is because the classes in XOR are not linearly separable. You cannot draw a straight line to separate the points (0,0),(1,1) from the points (0,1),(1,0). Led to invention of multi-layer networks.

8. Design a multi-layer perceptron to implement A XOR B.


```python
# importing Python library
import numpy as np
 
# define Unit Step Function
def unitStep(v):
    if v >= 0:
        return 1
    else:
        return 0
 
# design Perceptron Model
def perceptronModel(x, w, b):
    v = np.dot(w, x) + b
    y = unitStep(v)
    return y
 
# NOT Logic Function
# wNOT = -1, bNOT = 0.5
def NOT_logicFunction(x):
    wNOT = -1
    bNOT = 0.5
    return perceptronModel(x, wNOT, bNOT)
 
# AND Logic Function
# here w1 = wAND1 = 1, 
# w2 = wAND2 = 1, bAND = -1.5
def AND_logicFunction(x):
    w = np.array([1, 1])
    bAND = -1.5
    return perceptronModel(x, w, bAND)
 
# OR Logic Function
# w1 = 1, w2 = 1, bOR = -0.5
def OR_logicFunction(x):
    w = np.array([1, 1])
    bOR = -0.5
    return perceptronModel(x, w, bOR)
 
# XOR Logic Function
# with AND, OR and NOT  
# function calls in sequence
def XOR_logicFunction(x):
    y1 = AND_logicFunction(x)
    y2 = OR_logicFunction(x)
    y3 = NOT_logicFunction(y1)
    final_x = np.array([y2, y3])
    finalOutput = AND_logicFunction(final_x)
    return finalOutput
 
# testing the Perceptron Model
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])
 
print("XOR({}, {}) = {}".format(0, 1, XOR_logicFunction(test1)))
print("XOR({}, {}) = {}".format(1, 1, XOR_logicFunction(test2)))
print("XOR({}, {}) = {}".format(0, 0, XOR_logicFunction(test3)))
print("XOR({}, {}) = {}".format(1, 0, XOR_logicFunction(test4)))
```

    XOR(0, 1) = 1
    XOR(1, 1) = 0
    XOR(0, 0) = 0
    XOR(1, 0) = 1
    

9. Explain the single-layer feed forward architecture of ANN.

Feedforward neural networks are a powerful type of neural network that can be used for many things, such as recognising images and voices, processing natural languages, and making predictions. They process input data in a forward direction, from the input layer through one or more hidden layers, to produce an output from the output layer.

Single-layer feedforward networks, like the perceptron, have a single layer of input neurons connected to a single output neuron. On the other hand, multi-layer feedforward networks have one or more hidden layers of neurons between the input and output layers. This lets them learn more abstract features of the input data.

10. Explain the competitive network architecture of ANN.

Some of the most common architectures include: 

Feedforward Neural Networks: This is the simplest type of ANN architecture, where the information flows in one direction from input to output. The layers are fully connected, meaning each neuron in a layer is connected to all the neurons in the next layer.

Recurrent Neural Networks (RNNs): These networks have a “memory” component, where information can flow in cycles through the network. This allows the network to process sequences of data, such as time series or speech.

Convolutional Neural Networks (CNNs): These networks are designed to process data with a grid-like topology, such as images. The layers consist of convolutional layers, which learn to detect specific features in the data, and pooling layers, which reduce the spatial dimensions of the data.

Autoencoders: These are neural networks that are used for unsupervised learning. They consist of an encoder that maps the input data to a lower-dimensional representation and a decoder that maps the representation back to the original data.

Generative Adversarial Networks (GANs): These are neural networks that are used for generative modeling. They consist of two parts: a generator that learns to generate new data samples, and a discriminator that learns to distinguish between real and generated data.

11. Consider a multi-layer feed forward neural network. Enumerate and explain steps in the backpropagation algorithm used to train the network.
Training Algorithm :

Step 1: Initialize weight to small random values.

Step 2: While the stepsstopping condition is to be false do step 3 to 10.

Step 3: For each training pair do step 4 to 9 (Feed-Forward).

Step 4: Each input unit receives the signal unit and transmitsthe signal xi signal to all the units.

Step 5 : Each hidden unit Zj (z=1 to a) sums its weighted input signal to calculate its net input 
                  zinj = v0j + Σxivij     ( i=1 to n)

    Applying activation function zj = f(zinj) and sends this signals to all units in the layer about i.e output units

     For each output l=unit yk = (k=1 to m) sums its weighted input signals.

                     yink = w0k + Σ ziwjk    (j=1 to a)

     and applies its activation function to calculate the output signals.

                     yk = f(yink)

12. What are the advantages and disadvantages of neural networks?

Advantages:

Ability to handle complex data

Non-linear modeling capabilities

Adaptability and learning capabilities

Robustness to noisy or incomplete data	

Feature extraction capabilities	Potential

Domain agnostic, applicable to various business areas

Parallel processing for efficient computation

Can handle high-dimensional data

Can uncover hidden patterns and insights

Disadvantages:

Need for large amounts of labeled training data

Computationally intensive and resource-consuming

Lack of transparency in decision-making

overfitting without proper regularization

Complexity and difficulty in model tuning

Difficulty in explaining results to stakeholders

Sensitivity to input data quality and preprocessing

Ethical considerations in sensitive decision-making

13. Write short notes on any two of the following:

   1. Biological neuron
   2. ReLU function
   3. Single-layer feed forward ANN
   4. Gradient descent
   5. Recurrent networks
   
1. Biological neuron

Biological neuron models, also known as spiking neuron models, are mathematical descriptions of the conduction of electrical signals in neurons. Neurons (or nerve cells) are electrically excitable cells within the nervous system, able to fire electric signals, called action potentials, across a neural network. These mathematical models describe the role of the biophysical and geometrical characteristics of neurons on the conduction of electrical activity.

2. ReLU function

The rectified linear unit (ReLU) or rectifier activation function introduces the property of nonlinearity to a deep learning model and solves the vanishing gradients issue. It interprets the positive part of its argument. It is one of the most popular activation functions in deep learning. 

The activation function of a node defines the output of that node given an input or set of inputs. A standard integrated circuit can be seen as a digital network of activation functions that can be “ON” or “OFF,” depending on the input.
