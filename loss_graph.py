import matplotlib.pyplot as plt
import numpy as py
from calculate_cost import loss_function, expected_outputs
from layer import enhanced_forward_pass, layer1, input_activations


#weight_values = [-0.50, -0.35, -0.20, -0.05, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
weight_values = [-5.0, -3.5, -2.0, -0.5, 1.0, 2.5, 4.0, 5.5, 7.0, 8.5]
loss_change = []
for w in weight_values:
    layer1.neurons[1].weights[2] = w
    guessed_outputs = enhanced_forward_pass(layer1, input_activations)
    cost = loss_function(expected_outputs, guessed_outputs)
    loss_change.append(cost)
print(f"Change of loss {loss_change}")
print(f"Weight values {weight_values}")

plt.style.use("fivethirtyeight")
plt.plot(weight_values, loss_change, marker='o')
plt.xlabel("weight change")
plt.ylabel("loss change")
plt.title("Loss Graph")
plt.legend()
plt.show()