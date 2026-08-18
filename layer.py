from forward_pass import forward_pass
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Layer:
    def __init__(self):
        self.neurons = []
    def add(self, neuron):
        self.neurons.append(neuron)

neuron1 = Neuron([0.2, 0.7, 0.34], 0.12)
neuron2 = Neuron([0.8, 0.13, 0.52], 0.54)
neuron3 = Neuron([0.96, 0.64, 0.24], 0.72)

layer1 = Layer()

layer1.add(neuron1)
layer1.add(neuron2)
layer1.add(neuron3)

input_activations = [0.24, 0.654, 0.35]

def enhanced_forward_pass(layer, activations):
    num_neuron = len(layer.neurons)
    new_activations = []
    num = 0
    for i in range(num_neuron):
        curr_neuron = layer.neurons[i]
        num = forward_pass(curr_neuron.bias, curr_neuron.weights, activations)
        new_activations.append(num)
    return new_activations
print(enhanced_forward_pass(layer1, input_activations))