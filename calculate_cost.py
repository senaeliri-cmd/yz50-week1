from layer import enhanced_forward_pass, layer1, input_activations

expected_outputs = [0, 0, 1]

guessed_outputs = enhanced_forward_pass(layer1, input_activations)
total_loss_change_list = []

def loss_function(expected_outputs, guessed_outputs):
    cost = 0
    num_of_output = len(expected_outputs)

    for i in range(num_of_output):
        cost += (guessed_outputs[i] - expected_outputs[i])**2
    return cost

print(loss_function(expected_outputs, guessed_outputs))

neuron2_weight2_list = [0.89, 0.52, 0.11]
total_loss_change_list = [1.0665517963553084, 1.029837217771226, 0.9880886177233045]