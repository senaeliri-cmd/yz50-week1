weights = [0.2, 0.02, 0.7]
a = [0.5, 0.2, 0.1]
bias = 0.7
E = 2.718281828459045

def forward_pass(bias, weights, a):
    sum = 0.0
    length = len(weights)
    for i in range(length):
        sum += a[i]*weights[i]
    sum += bias

    return sigmoid(sum)


def sigmoid(x):
    return 1 / (1 + (1/E**x))

