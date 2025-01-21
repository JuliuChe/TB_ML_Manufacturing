import numpy as np
import nnfs #nnfs is for simulation of a dataset of inputs
from nnfs.datasets import spiral_data
nnfs.init() #Setup a default datatype

X, y = spiral_data(100,3)


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


layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
