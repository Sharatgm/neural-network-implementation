import random
import math
from csv import reader
import numpy as np
#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer
#   neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # normalize
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    return dataset


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split data into train and test
def split_train_test(x, train_portion):
    train_num_rows = round(train_portion * (len(x)))
    # pick your indices for sample 1 and sample 2:
    s1 = np.random.choice(range(x.shape[0]), train_num_rows, replace=False)
    s2 = list(set(range(x.shape[0])) - set(s1))
    # extract samples:
    train = x[s1]
    test = x[s2]

    # Format train set
    f_train = []
    for row in train:
        r = []
        row_li = row.tolist()
        r.append(row_li[0:-1])
        r.append([row_li[-1]])
        f_train.append(r)

    # Format test set
    f_test = []
    for row in test:
        r = []
        row_li = row.tolist()
        r.append(row_li[0:-1])
        r.append([row_li[-1]])
        f_test.append(r)

    return f_train, f_test


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


class NeuralNetwork:
    def __init__(self, learning_rate, num_inputs, num_hidden_neurons, num_outputs, num_hidden_layers, hidden_layer_weights=None,
                 hidden_layer_bias=None, output_layer_weights=None, output_layer_bias=None):
        self.learning_rate = learning_rate
        self.num_inputs = num_inputs
        self.hidden_layers = []

        for i in range(0,num_hidden_layers):
            self.hidden_layers.append(NeuronLayer(num_hidden_neurons, hidden_layer_bias))
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_hidden_layer_neurons(hidden_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layers[0].neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layers[0].neurons[h].weights.append(random.random())
                else:
                    self.hidden_layers[0].neurons[h].weights.append(hidden_layer_weights[0][weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_hidden_layer_neurons(self, hidden_layer_weights):
        for l in range(1, len(self.hidden_layers)):
            for h in range(len(self.hidden_layers[l].neurons)):
                for i in range(len(self.hidden_layers[l-1].neurons)):
                    if not hidden_layer_weights:
                        self.hidden_layers[l].neurons[h].weights.append(random.random())
                    else:
                        self.hidden_layers[l].neurons[h].weights.append(hidden_layer_weights[l][i])


    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layers[-1].neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        for i in range(len(self.hidden_layers)):
            print('Hidden Layer', i)
            self.hidden_layers[i].inspect()
            print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        for layer in self.hidden_layers:
            inputs = layer.feed_forward(inputs)
        return self.output_layer.feed_forward(inputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        #print('Output:', self.output_layer.get_outputs())
        # 1. Output neuron deltas
        deltas_output_layer = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            deltas_output_layer[o] = self.output_layer.neurons[o].calculate_delta(training_outputs[o])

        # 2. Hidden neuron deltas
        deltas_hidden_layer = [[0] * len(self.hidden_layers[0].neurons)] * len(self.hidden_layers)
        for l in range(len(self.hidden_layers)-1,0):
            for h in range(len(self.hidden_layers[l].neurons)):

                # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                d_error_wrt_hidden_neuron_output = 0
                if (l == len(self.hidden_layers[l])-1): # Use the output to get the deltas of last hidden layer
                    for o in range(len(self.output_layer.neurons)):
                        d_error_wrt_hidden_neuron_output += deltas_output_layer[o] * self.output_layer.neurons[o].weights[h]
                else:
                    for o in range(len(self.hidden_layers[l+1].neurons)):
                        d_error_wrt_hidden_neuron_output += deltas_hidden_layer[l+1][o] * self.hidden_layers[l+1].neurons[o].weights[h]
                # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                deltas_hidden_layer[l][h] = d_error_wrt_hidden_neuron_output * self.hidden_layers[l].neurons[h].calculate_pd_logistic_function()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                gradient_wrt_weight = deltas_output_layer[o] * self.output_layer.neurons[o].input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.learning_rate * gradient_wrt_weight

        # 4. Update hidden neuron weights
        for l in range(len(self.hidden_layers)):
            for h in range(len(self.hidden_layers[l].neurons)):
                for w_ih in range(len(self.hidden_layers[l].neurons[h].weights)):

                    # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                    gradient_wrt_weight = deltas_hidden_layer[l][h] * self.hidden_layers[l].neurons[h].input_wrt_weight(w_ih)

                    # Δw = α * ∂Eⱼ/∂wᵢ
                    self.hidden_layers[l].neurons[h].weights[w_ih] -= self.learning_rate * gradient_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error_mean_square(training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias, activation_fun=Neuron.sigmoid))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    def __init__(self, bias, activation_fun=None):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = 0
        self.activation_fun = activation_fun or self.sigmoid

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.activate(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def activate(self, total_net_input):
        return self.activation_fun(total_net_input)

    # Apply the logistic function to squash the output of the neuron
    @staticmethod
    def sigmoid(total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Apply the relu function to squash the output of the neuron
    @staticmethod
    def relu(total_net_input):
        return max(0, total_net_input)

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_delta(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_logistic_function()

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error_mean_square(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_logistic_function(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def input_wrt_weight(self, index):
        return self.inputs[index]


# predict y values for a complete set
def predict_set(nn, dataset):
    actual = []
    predicted = []
    for i in range(0, len(dataset)):
        y_hat = nn.feed_forward(dataset[i][0])
        actual.append(dataset[i][1][0])
        predicted.append(round(y_hat[0]))
    return actual, predicted


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def main():
    train_size = 0.8
    l_rate = 0.03
    num_epoch = 200000

    # load and prepare data
    filename = 'pima-indians-diabetes.csv'
    dataset = np.array(load_csv(filename))
    print("\ndataset.shape:")
    print(dataset.shape)

    # split and format train and test set from dataset
    train, test = split_train_test(dataset, train_size)

    nn = NeuralNetwork(l_rate, len(train[0][0]), 20, len(train[0][1]), 1)
    for i in range(num_epoch):
        training_inputs, training_outputs = random.choice(train)
        nn.train(training_inputs, training_outputs)

    print("\nTrain set error:")
    print(nn.calculate_total_error(train))

    print("\nTrain set accuracy:")
    actual, predicted = predict_set(nn, train)
    print(str(round(accuracy_metric(actual, predicted), 2)) + "%")

    print("\n\nTest set error:")
    print(nn.calculate_total_error(test))

    print("\nTest set accuracy:")
    actual, predicted = predict_set(nn, test)
    print(str(round(accuracy_metric(actual, predicted), 2)) + "%")

    # for row in test:
        # print(nn.feed_forward(row[0]))

    print("\nSingle test: model output vs predicted")
    print(nn.feed_forward(test[0][0]))
    print(test[0][1])

    ##
    #Blog post example:
    # nn = NeuralNetwork(learning_rate=l_rate, num_inputs=2, num_hidden=2, num_outputs=2,
    #                    hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35,
    #                    output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
    # for i in range(num_epoch):
    #     nn.train([0.05, 0.1], [0.01, 0.99])
    #     print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))
    #
    # #XOR example:

    # training_sets = [
    #     [[0, 0], [0]],
    #     [[0, 1], [1]],
    #     [[1, 0], [1]],
    #     [[1, 1], [0]]
    # ]
    #
    # nn = NeuralNetwork(l_rate, len(training_sets[0][0]), 5, len(training_sets[0][1]), 1)
    # nn.inspect()
    # for i in range(num_epoch):
    #     training_inputs, training_outputs = random.choice(training_sets)
    #     nn.train(training_inputs, training_outputs)
    #     #print('Expected: ', training_outputs)
    #     print(i, nn.calculate_total_error(training_sets))


if __name__ == "__main__":
    main()
