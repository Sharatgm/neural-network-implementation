import random
import math
from csv import reader
import numpy as np
from random import randrange
import matplotlib.pylab as plt

# global variable to store the errors/loss for visualisation
__errors__ = []


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
    def __init__(self, learning_rate, num_inputs, num_hidden_neurons, num_outputs, num_hidden_layers,
                 reg_factor_over_n=0.0, hidden_layer_weights=None,
                 hidden_layer_bias=None, output_layer_weights=None, output_layer_bias=None):
        self.learning_rate = learning_rate
        self.num_inputs = num_inputs
        self.hidden_layers = []

        self.reg_factor_over_n = reg_factor_over_n

        for i in range(0, num_hidden_layers):
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
                for i in range(len(self.hidden_layers[l - 1].neurons)):
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

        # 1. Output neuron deltas
        deltas_output_layer = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            deltas_output_layer[o] = self.output_layer.neurons[o].calculate_delta(training_outputs[o])

        # 2. Hidden neuron deltas
        deltas_hidden_layer = [[0] * len(self.hidden_layers[0].neurons)] * len(self.hidden_layers)
        for l in range(len(self.hidden_layers) - 1, 0):
            for h in range(len(self.hidden_layers[l].neurons)):

                d_error_wrt_hidden_neuron_output = 0
                if (l == len(self.hidden_layers[l]) - 1):  # Use the output to get the deltas of last hidden layer
                    for o in range(len(self.output_layer.neurons)):
                        d_error_wrt_hidden_neuron_output += deltas_output_layer[o] * \
                                                            self.output_layer.neurons[o].weights[h]
                else:
                    for o in range(len(self.hidden_layers[l + 1].neurons)):
                        d_error_wrt_hidden_neuron_output += deltas_hidden_layer[l + 1][o] * \
                                                            self.hidden_layers[l + 1].neurons[o].weights[h]

                deltas_hidden_layer[l][h] = d_error_wrt_hidden_neuron_output * self.hidden_layers[l].neurons[
                    h].calculate_pd_logistic_function()

        # Check if regulation is needed
        if (self.reg_factor_over_n != 0):  # With regularization

            # 3. Update output neuron weights
            for o in range(len(self.output_layer.neurons)):
                for w_ho in range(len(self.output_layer.neurons[o].weights)):
                    # Calculate the gradient of the weight
                    gradient_wrt_weight = deltas_output_layer[o] * self.output_layer.neurons[o].input_wrt_weight(w_ho)

                    # Update weight with regularization -> newW = (1 - l_rate * (reg_factor / n_samples)) * weight - l_rate * weight
                    self.output_layer.neurons[o].weights[w_ho] = (1 - self.learning_rate * self.reg_factor_over_n) * \
                                                                 self.output_layer.neurons[o].weights[
                                                                     w_ho] - self.learning_rate * gradient_wrt_weight

            # 4. Update hidden neuron weights
            for l in range(len(self.hidden_layers)):
                for h in range(len(self.hidden_layers[l].neurons)):
                    for w_ih in range(len(self.hidden_layers[l].neurons[h].weights)):
                        # Calculate the gradient -> a * delta
                        gradient_wrt_weight = self.hidden_layers[l].neurons[h].input_wrt_weight(w_ih) * \
                                              deltas_hidden_layer[l][h]

                        # Update weight with regularization -> newW = (1 - l_rate * (reg_factor / n_samples)) * weight - l_rate * weight
                        self.hidden_layers[l].neurons[h].weights[w_ih] = (
                                                                                     1 - self.learning_rate * self.reg_factor_over_n) * \
                                                                         self.hidden_layers[l].neurons[h].weights[
                                                                             w_ih] - self.learning_rate * gradient_wrt_weight

        else:  # Without regularization

            # 3. Update output neuron weights
            for o in range(len(self.output_layer.neurons)):
                for w_ho in range(len(self.output_layer.neurons[o].weights)):
                    # Calculate the gradient of the weight
                    gradient_wrt_weight = deltas_output_layer[o] * self.output_layer.neurons[o].input_wrt_weight(w_ho)

                    # Update weight without regularization -> newW = weight - l_rate * gradient
                    self.output_layer.neurons[o].weights[w_ho] -= self.learning_rate * gradient_wrt_weight

            # 4. Update hidden neuron weights
            for l in range(len(self.hidden_layers)):
                for h in range(len(self.hidden_layers[l].neurons)):
                    for w_ih in range(len(self.hidden_layers[l].neurons[h].weights)):
                        # Calculate the gradient -> a * delta
                        gradient_wrt_weight = deltas_hidden_layer[l][h] * self.hidden_layers[l].neurons[
                            h].input_wrt_weight(w_ih)

                        # Update weight without regularization -> newW = weight - l_rate * gradient
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

    # Function to calculate deltas
    def calculate_delta(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_logistic_function()

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error_mean_square(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # Partial derivative error of each neuron used to calculate the deltas
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The partial derivative of the logistic function used to calculate the deltas -> yⱼ * (1 - yⱼ)
    def calculate_pd_logistic_function(self):
        return self.output * (1 - self.output)

    # Returns the input with respect the weights to calculate the gradient (a)
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


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def cross_validate(train, n_folds, regularization_factor, l_rate, n_hidden_layers, n_neurons, num_epoch):
    folds = cross_validation_split(train, n_folds)
    scores = list()
    for fold in folds:
        fold_train_set = list(folds)
        fold_train_set.remove(fold)
        fold_test_set = list(fold)
        reg_factor_over_n = regularization_factor / len(fold_test_set)
        nn = NeuralNetwork(l_rate, len(fold_train_set[0][0][0]), n_neurons, len(fold_train_set[0][0][1]), n_hidden_layers,
                           reg_factor_over_n)
        for i in range(num_epoch):
            for fold_i in range(0, len(fold_train_set)) :
                for j in range(0, len(fold_train_set[fold_i])):
                    training_input, training_output = fold_train_set[fold_i][j]
                    nn.train(training_input, training_output)
        actual, predicted = predict_set(nn, fold_test_set)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def main():
    train_size = 0.9
    n_folds = 5
    l_rate = 0.003
    regularization_factor = 0
    n_hidden_layers = 3
    n_neurons = 16
    num_epoch = 10000

    # load and prepare data
    filename = 'pima-indians-diabetes.csv'
    dataset = np.array(load_csv(filename))
    print("\ndataset.shape:")
    print(dataset.shape)

    # split and format train and test set from dataset
    train, test = split_train_test(dataset, train_size)

    #scores = cross_validate(train, n_folds, regularization_factor, l_rate, n_hidden_layers, n_neurons, num_epoch)
    #print(scores)
    #print(sum(scores) / len(scores))

    reg_factor_over_n = regularization_factor / len(train)
    nn = NeuralNetwork(l_rate, len(train[0][0]), n_neurons, len(train[0][1]), n_hidden_layers, reg_factor_over_n)
    for epoch in range(num_epoch):
        for j in range(len(train)):
            training_input = train[j][0]
            training_output = train[j][1]
            nn.train(training_input, training_output)
        __errors__.append(nn.calculate_total_error(train))
        if epoch > 2:
            if __errors__[epoch-1] > (__errors__[epoch-2] + 1):
                print("\nDiverging")
                break
        if epoch % 100 == 0:
            print(epoch)
            print(__errors__[epoch-1])

    print("\nTrain set error:")
    train_error = nn.calculate_total_error(train)
    print(train_error)

    print("\nTrain set accuracy:")
    actual, predicted = predict_set(nn, train)
    print(str(round(accuracy_metric(actual, predicted), 2)) + "%")

    print("\n\nTest set error:")
    test_error = nn.calculate_total_error(test)
    print(test_error)

    print("\nTest set accuracy:")
    actual, predicted = predict_set(nn, test)
    print(str(round(accuracy_metric(actual, predicted), 2)) + "%")

    print("\nModel variance:")
    print(str(round(test_error - train_error)) + "%")

    # for row in test:
    # print(nn.feed_forward(row[0]))

    print("\nSingle test: model output vs predicted")
    print(nn.feed_forward(test[0][0]))
    print(test[0][1])

    # plot square mean error
    num_epochs_error = range(1, len(__errors__) + 1)
    plt.plot(num_epochs_error, __errors__)
    plt.show()

    ##
    # Blog post example:
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
