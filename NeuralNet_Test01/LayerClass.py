import numpy as np

#X is our input data : 3 samples
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, i))
print(output)

#Layer class takes inputs and number of neurons
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases



layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer1.output)
print(layer2.output)
