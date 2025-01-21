import numpy as np
import nnfs #nnfs is for simulation of a dataset of inputs
from nnfs.datasets import spiral_data
nnfs.init() #Setup a default datatype




class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

#Activation function Rectified Linear Units
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(inputs, 0)

class ActivationSoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(100,3)
dense1 = LayerDense(2,3)
activation1 = ActivationReLU()

dense2=LayerDense(3, 3)
activation2=ActivationSoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])











'''
#Softmax principle outside a class with numpy
import numpy as np
layer_outputs=[[4.8,1.21, 2.385],
               [8.9,-1.81,0.2],
               [1.41,1.051,0.026]]

exp_values = np.exp(layer_outputs)
print(exp_values)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)
print(np.sum(layer_outputs, axis=1, keepdims=True))
'''
'''
#With Math library only
import math
E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)


print(exp_values)

norm_base=sum(exp_values)
norm_values = []

for val in exp_values:
    norm_values.append(val/norm_base)

print(norm_values)
'''