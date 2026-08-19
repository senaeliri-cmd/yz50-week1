import matplotlib.pyplot as plt
import numpy as py
import math
from calculate_cost import loss_function, expected_outputs, guessed_outputs
from layer import enhanced_forward_pass, layer1, input_activations

h = 0.001

def numerical_derivative(weight):
    layer1.neurons[1].weights[2] = weight
    guessed_outputs = enhanced_forward_pass(layer1, input_activations)

    layer1.neurons[1].weights[2] = weight + h
    guessed_outputs_new = enhanced_forward_pass(layer1, input_activations)

    layer1.neurons[1].weights[2] = weight

    return (loss_function(expected_outputs, guessed_outputs_new)- loss_function(expected_outputs, guessed_outputs))/ h
n = 0
x = []
y = []
a = 0.7
learning_rate = 0.2
tolerance = 0.01
gradient = numerical_derivative(a)

while(abs(gradient) > tolerance and n < 1000):
    step = learning_rate * gradient
    a = a - step
    gradient = numerical_derivative(a)

    layer1.neurons[1].weights[2] = a
    learning_guess = enhanced_forward_pass(layer1, input_activations)
    loss = loss_function(expected_outputs, learning_guess)

    y.append(loss)
    x.append(n)
    n += 1

    print(f"updated gradient= {gradient} weight = {a} loss = {loss}")

plt.plot(x, y, marker='o')
plt.title("")
plt.show()