import numpy as np

#First neuron
inputs = [1, 2, 3, 2.5]
weights1=[0.2, 0.8, -0.5, 1.0]
bias1 = 2

output1 = inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bias1
print(output1)

#First layer - 3 neurons
weights2=[0.5, -0.91, 0.26, -0.5]
weights3=[-0.26, -0.27, 0.17, 0.87]
bias2 = 3
bias3 = 0.5

output2 = inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2
output3 = inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3
output=[output1, output2, output3]
print(output)


#Simplified 1st layer - For loops
#First neuron
inputs = [1, 2, 3, 2.5]
weights=[weights1,weights2,weights3]
biases = [bias1,bias2,bias3]
layer_output=[]
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output =0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output+=n_input*weight
    neuron_output+=neuron_bias
    layer_output.append(neuron_output)

print(layer_output)

#Simplified1st Layer - Using dot products - Batch of inputs
inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
layer1Outputs = np.dot(inputs, np.array(weights).T)+(biases)
print(layer1Outputs)

#Adding a second layer
weights2=[[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]]
biases2=[-1,2,-0.5]
layer2Outputs = np.dot(layer1Outputs, np.array(weights2).T)+(biases2)
print(layer2Outputs)

#Let's get into an object

