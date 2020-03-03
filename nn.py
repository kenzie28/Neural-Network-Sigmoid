from scipy.special import expit as sigmoid
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

# Function used to go through a matrice passed as a parameter and randomize the values
def random_initialize(matrice, epsilon=0.12):
    # randomize the initialization
    for row in range(len(matrice)):
        for col in range(len(matrice[row])):            
            matrice[row][col] = matrice[row][col] * 2 * epsilon - epsilon

# Load the dataset
filename = input("Input file name: ")
print("Loading dataset...")
data = pd.read_csv(filename)

# Transforms the data to numpy arrays and splits it into train and test sets
# Note that the labeled data should be in the first column
features = data.values[:, 1:]
output = data.values[:, 0]
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3)

X_features = x_train
y_label = y_train

m = len(X_features)
num_features = len(X_features[0])

# Converts the y_labels to a matrix where the index number 
# in a row that has the value 1 shows the correct output
y = np.zeros((len(y_label), len(np.unique(y_label))))
for i in range(len(y_label)):
    y[i][y_label[i]] = 1

# Generates the hidden layers of the neural network
num_layers = 0
while num_layers <= 0:
    num_layers = int(input("Number of hidden layers to use: "))

network = np.empty(1 + num_layers, dtype=np.ndarray)

num_nodes = int(input("Number of nodes to use for layer 1: "))

# Generates the first layer and stores it to the network
first_layer = np.random.rand(num_nodes, num_features + 1)
random_initialize(first_layer)
network[0] = first_layer

# Asks for the number of nodes to use for each layer and creates the layer accordingly
# The layer is created to deal with a bias unit. 
# All layers will be randomly initialized and stored in network
for i in range(num_layers - 1):
    num_nodes = int(input("Number of nodes to use for layer " + str(i + 2) + ": "))
    layer = np.random.rand(num_nodes, len(network[i]) + 1)
    random_initialize(layer)
    network[i + 1] = layer

num_labels = len(np.unique(y_label))

# Creates the final layer of output nodes
output_nodes = np.random.rand(num_labels, len(network[-2]) + 1)
random_initialize(output_nodes)
network[-1] = output_nodes

print("Network Shape:")

for layer in network:
    print(layer.shape)

print("Starting backpropagation...")

# Start the hard part
reg_term = float(input("Input regularization parameter (0 to ignore regularization): "))

# Creates a list of gradients to hold the obtained values
gradients = []
for i in range(len(network)):
    gradients.append(np.zeros(network[i].shape))

training_steps = 300

for n in range(training_steps):
    print("Training iteration: " + str(n + 1))
    percent = 0
    J = 0
    for i in range(m):
        if i % (m // 10) == 0:
            print(str(percent) + "%")
            percent += 10

        # FORWARD PROPOGATION

        # Create list to store calculated values
        calculated_layers = []

        # Column vector of one sample of X_features 
        # with a bias unit added on top.
        X = np.append(np.array([1]), X_features[i], axis=0)
        
        theta1 = network[0]

        # Calculates the output of the first layer activating
        # the X features and stores the results to calculated_layers
        first_output = sigmoid(np.dot(theta1, X))
        first_output = np.append(np.array([1]), first_output, axis=0)
        first_output = np.reshape(first_output, (-1, 1))
        calculated_layers.append(first_output)

        # Processes all the outputs for the layers except the last one
        for j in range(num_layers):
            # Gets the previous output and the corresponding network layer
            values = calculated_layers[j]
            theta = network[j + 1]

            # Calculates the new output, append a bias unit, reshape it to a column vector
            result = sigmoid(np.dot(theta, values)).T
            result = np.append(np.array([1]), result[0])
            result = np.reshape(result, (-1, 1))

            calculated_layers.append(result)

        # Obtains the output, removes the bias unit and reshape it to a row vector
        result = calculated_layers.pop(-1)[1:]
        result = np.reshape(result, (1, -1))

        J += (y[i] * np.log(result)) - (np.ones(y[i].shape) - y[i]) * \
            np.log(np.ones(result.shape) - result)

        # Initialize empty list to hold delta values
        delta_layers = []

        # Calculate the last data value and append it to list
        last_delta = result - y[i]
        delta_layers.append(last_delta)

        result = np.where(result == np.amax(result))

        # Calculates the delta values in reverse order
        for j in range(num_layers):
            # Gets the previous delta value, the theta and the derivative 
            # of the activated output at the corresponding layer
            prev_delta = delta_layers[j]
            theta = network[-1 - j]
            output = calculated_layers[-1 - j]
            derivative = output * (np.ones(output.shape) - output)

            # Calculates the delta value, deletes the bias unit, 
            # reshapes it to a row vector
            delta = np.dot(prev_delta, theta).T * derivative
            delta = np.delete(delta, 0, axis=0)
            delta = np.reshape(delta, (1, -1))
            
            delta_layers.append(delta)

        # Reverses the order of the delta_layers
        delta_layers.reverse()

        # Calculates all the gradients except for the second layer
        for j in range(1, len(gradients)):
            gradients[j] += np.dot(calculated_layers[j - 1], delta_layers[j]).T

        # Calculates the gradient for second layer
        X = np.reshape(X, (-1, 1))
        gradients[0] = np.dot(X, delta_layers[0]).T

    # Prints out cost function
    J /= m
    print(np.sum(J))

    # Updates the network with gradients (regularized)
    for i in range(len(gradients)):
        gradients[i] /= m
        network[i] -= gradients[i] * (1.01**(training_steps - n)) + (reg_term / (2 * m)) * network[i]

print("Backpropogation Complete")

for layer in network:
    print(layer.shape)

X_features = x_test

y_label = y_test

training_samples = m
m = len(X_features)
num_features = len(X_features[0])

layer = []
layer.append(X_features)

for i in range(len(network)):
    input_layer = layer[i]
    parameter = network[i]

    input_layer = np.append(np.ones((len(X_features), 1)), input_layer, axis=1).T

    output = np.dot(parameter, input_layer).T
    print(output.shape)
    layer.append(output)
    
result = layer[-1]

predictions = []
for i in range(len(result)):
    predict = 0
    highest_prob = 0
    for j in range(len(result[i])):
        if result[i][j] > highest_prob:
            predict = j
            highest_prob = result[i][j]
    predictions.append(predict)

total_labels = len(predictions)
correct = 0
for i in range(len(predictions)):
    print(str(i) + ": Predicted: " + str(predictions[i]) + "; Actual: " + str(y_label[i]))
    if predictions[i] == y_label[i]:
        correct += 1

print("Achieved " + str(math.ceil(correct / total_labels * 100)) + "% accuracy")
print("Using " + str(training_samples) + " training examples")
print("With network shape: ")
for layer in network:
    print(layer.shape)
