from math import exp
from random import random
from copy import deepcopy
from random import random, seed


# Иницијализација на мрежа
# Ставете фиксни тежини од 0.5 на code.finki.ukim.mk ако постои проблем со random()
def initialize_network(n_inputs, n_hidden, n_outputs):
    """Изградба на мрежата и иницијализација на тежините

    :param n_inputs: број на неврони во влезниот слој
    :type n_inputs: int
    :param n_hidden: број на неврони во скриениот слој
    :type n_hidden: int
    :param n_outputs: број на неврони во излезниот слој
                      (број на класи)
    :type n_outputs: int
    :return: мрежата како листа на слоеви, каде што секој
             слој е речник со клуч 'weights' и нивните вредности
    :rtype: list(list(dict(str, list)))
    """
    network = list()
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]}
                    for _ in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]}
                    for _ in range(n_outputs)]
    network.append(output_layer)
    return network


def neuron_calculate(weights, inputs):
    """Пресметување на вредноста за активација на неврон

    :param weights: даден вектор (листа) на тежини
    :type weights: list(float)
    :param inputs: даден вектор (листа) на влезови
    :type inputs: list(float)
    :return: пресметка на невронот
    :rtype: float
    """
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def sigmoid_activation(activation):
    """Sigmoid активациска функција

    :param activation: вредност за активациската функција
    :type activation: float
    :return: вредност добиена од примена на активациската
             функција
    :rtype: float
    """
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    """Пропагирање нанапред на влезот кон излезот на мрежата

    :param network: дадената мрежа
    :param row: моменталната податочна инстаца
    :return: листа на излезите од последниот слој
    """
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron_calculate(neuron['weights'], inputs)
            neuron['output'] = sigmoid_activation(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def sigmoid_activation_derivative(output):
    """Пресметување на изводот на излезот од невронот

    :param output: излезни вредности
    :return: вредност на изводот
    """
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    """Пропагирање на грешката наназад и сочувување во невроните

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param expected: очекувани вредности за излезот
    :type expected: list(int)
    :return: None
    """
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_activation_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    """Ажурирање на тежините на мрежата со грешката

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param row: една инстанца на податоци
    :type row: list
    :param l_rate: рата на учење
    :type l_rate: float
    :return: None
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs, verbose=True):
    """Тренирање на мрежата за фиксен број на епохи

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param train: тренирачко множество
    :type train: list
    :param l_rate: рата на учење
    :type l_rate: float
    :param n_epoch: број на епохи
    :type n_epoch: int
    :param n_outputs: број на неврони (класи) во излезниот слој
    :type n_outputs: int
    :param verbose: True за принтање на лог, инаку False
    :type: verbose: bool
    :return: None
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0] * n_outputs
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        if verbose:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    """Направи предвидување

    :param network: дадена мрежа
    :type network: list(list(dict(str, list)))
    :param row: една податочна инстанца
    :type row: list
    :return: предвидени класи
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

seed(1)
training_data = [
    [3.6216, 8.6661, -2.8073, -0.44699, 0],
    [4.5459, 8.1674, -2.4586, -1.4621, 0],
    [3.866, -2.6383, 1.9242, 0.10645, 0],
    [3.4566, 9.5228, -4.0112, -3.5944, 0],
    [0.32924, -4.4552, 4.5718, -0.9888, 0],
    [4.3684, 9.6718, -3.9606, -3.1625, 0],
    [3.5912, 3.0129, 0.72888, 0.56421, 0],
    [2.0922, -6.81, 8.4636, -0.60216, 0],
    [3.2032, 5.7588, -0.75345, -0.61251, 0],
    [1.5356, 9.1772, -2.2718, -0.73535, 0],
    [1.2247, 8.7779, -2.2135, -0.80647, 0],
    [3.9899, -2.7066, 2.3946, 0.86291, 0],
    [1.8993, 7.6625, 0.15394, -3.1108, 0],
    [-1.5768, 10.843, 2.5462, -2.9362, 0],
    [3.404, 8.7261, -2.9915, -0.57242, 0],
    [4.6765, -3.3895, 3.4896, 1.4771, 0],
    [2.6719, 3.0646, 0.37158, 0.58619, 0],
    [0.80355, 2.8473, 4.3439, 0.6017, 0],
    [1.4479, -4.8794, 8.3428, -2.1086, 0],
    [5.2423, 11.0272, -4.353, -4.1013, 0],
    [5.7867, 7.8902, -2.6196, -0.48708, 0],
    [0.3292, -4.4552, 4.5718, -0.9888, 0],
    [3.9362, 10.1622, -3.8235, -4.0172, 0],
    [0.93584, 8.8855, -1.6831, -1.6599, 0],
    [4.4338, 9.887, -4.6795, -3.7483, 0],
    [0.7057, -5.4981, 8.3368, -2.8715, 0],
    [1.1432, -3.7413, 5.5777, -0.63578, 0],
    [-0.38214, 8.3909, 2.1624, -3.7405, 0],
    [6.5633, 9.8187, -4.4113, -3.2258, 0],
    [4.8906, -3.3584, 3.4202, 1.0905, 0],
    [-0.24811, -0.17797, 4.9068, 0.15429, 0],
    [1.4884, 3.6274, 3.308, 0.48921, 0],
    [4.2969, 7.617, -2.3874, -0.96164, 0],
    [-0.96511, 9.4111, 1.7305, -4.8629, 0],
    [-1.6162, 0.80908, 8.1628, 0.60817, 0],
    [2.4391, 6.4417, -0.80743, -0.69139, 0],
    [2.6881, 6.0195, -0.46641, -0.69268, 0],
    [3.6289, 0.81322, 1.6277, 0.77627, 0],
    [4.5679, 3.1929, -2.1055, 0.29653, 0],
    [3.4805, 9.7008, -3.7541, -3.4379, 0],
    [4.1711, 8.722, -3.0224, -0.59699, 0],
    [-0.2062, 9.2207, -3.7044, -6.8103, 0],
    [-0.0068919, 9.2931, -0.41243, -1.9638, 0],
    [0.96441, 5.8395, 2.3235, 0.066365, 0],
    [2.8561, 6.9176, -0.79372, 0.48403, 0],
    [-0.7869, 9.5663, -3.7867, -7.5034, 0],
    [2.0843, 6.6258, 0.48382, -2.2134, 0],
    [-0.7869, 9.5663, -3.7867, -7.5034, 0],
    [3.9102, 6.065, -2.4534, -0.68234, 0],
    [1.6349, 3.286, 2.8753, 0.087054, 0],
    [4.3239, -4.8835, 3.4356, -0.5776, 0],
    [5.262, 3.9834, -1.5572, 1.0103, 0],
    [3.1452, 5.825, -0.51439, -1.4944, 0],
    [2.549, 6.1499, -1.1605, -1.2371, 0],
    [4.9264, 5.496, -2.4774, -0.50648, 0],
    [4.8265, 0.80287, 1.6371, 1.1875, 0],
    [2.5635, 6.7769, -0.61979, 0.38576, 0],
    [5.807, 5.0097, -2.2384, 0.43878, 0],
    [3.1377, -4.1096, 4.5701, 0.98963, 0],
    [-0.78289, 11.3603, -0.37644, -7.0495, 0],
    [-1.3971, 3.3191, -1.3927, -1.9948, 1],
    [0.39012, -0.14279, -0.031994, 0.35084, 1],
    [-1.6677, -7.1535, 7.8929, 0.96765, 1],
    [-3.8483, -12.8047, 15.6824, -1.281, 1],
    [-3.5681, -8.213, 10.083, 0.96765, 1],
    [-2.2804, -0.30626, 1.3347, 1.3763, 1],
    [-1.7582, 2.7397, -2.5323, -2.234, 1],
    [-0.89409, 3.1991, -1.8219, -2.9452, 1],
    [0.3434, 0.12415, -0.28733, 0.14654, 1],
    [-0.9854, -6.661, 5.8245, 0.5461, 1],
    [-2.4115, -9.1359, 9.3444, -0.65259, 1],
    [-1.5252, -6.2534, 5.3524, 0.59912, 1],
    [-0.61442, -0.091058, -0.31818, 0.50214, 1],
    [-0.36506, 2.8928, -3.6461, -3.0603, 1],
    [-5.9034, 6.5679, 0.67661, -6.6797, 1],
    [-1.8215, 2.7521, -0.72261, -2.353, 1],
    [-0.77461, -1.8768, 2.4023, 1.1319, 1],
    [-1.8187, -9.0366, 9.0162, -0.12243, 1],
    [-3.5801, -12.9309, 13.1779, -2.5677, 1],
    [-1.8219, -6.8824, 5.4681, 0.057313, 1],
    [-0.3481, -0.38696, -0.47841, 0.62627, 1],
    [0.47368, 3.3605, -4.5064, -4.0431, 1],
    [-3.4083, 4.8587, -0.76888, -4.8668, 1],
    [-1.6662, -0.30005, 1.4238, 0.024986, 1],
    [-2.0962, -7.1059, 6.6188, -0.33708, 1],
    [-2.6685, -10.4519, 9.1139, -1.7323, 1],
    [-0.47465, -4.3496, 1.9901, 0.7517, 1],
    [1.0552, 1.1857, -2.6411, 0.11033, 1],
    [1.1644, 3.8095, -4.9408, -4.0909, 1],
    [-4.4779, 7.3708, -0.31218, -6.7754, 1],
    [-2.7338, 0.45523, 2.4391, 0.21766, 1],
    [-2.286, -5.4484, 5.8039, 0.88231, 1],
    [-1.6244, -6.3444, 4.6575, 0.16981, 1],
    [0.50813, 0.47799, -1.9804, 0.57714, 1],
    [1.6408, 4.2503, -4.9023, -2.6621, 1],
    [0.81583, 4.84, -5.2613, -6.0823, 1],
    [-5.4901, 9.1048, -0.38758, -5.9763, 1],
    [-3.2238, 2.7935, 0.32274, -0.86078, 1],
    [-2.0631, -1.5147, 1.219, 0.44524, 1],
    [-0.91318, -2.0113, -0.19565, 0.066365, 1],
    [0.6005, 1.9327, -3.2888, -0.32415, 1],
    [0.91315, 3.3377, -4.0557, -1.6741, 1],
    [-0.28015, 3.0729, -3.3857, -2.9155, 1],
    [-3.6085, 3.3253, -0.51954, -3.5737, 1],
    [-6.2003, 8.6806, 0.0091344, -3.703, 1],
    [-4.2932, 3.3419, 0.77258, -0.99785, 1],
    [-3.0265, -0.062088, 0.68604, -0.055186, 1],
    [-1.7015, -0.010356, -0.99337, -0.53104, 1],
    [-0.64326, 2.4748, -2.9452, -1.0276, 1],
    [-0.86339, 1.9348, -2.3729, -1.0897, 1],
    [-2.0659, 1.0512, -0.46298, -1.0974, 1],
    [-2.1333, 1.5685, -0.084261, -1.7453, 1],
    [-1.2568, -1.4733, 2.8718, 0.44653, 1],
    [-3.1128, -6.841, 10.7402, -1.0172, 1],
    [-4.8554, -5.9037, 10.9818, -0.82199, 1],
    [-2.588, 3.8654, -0.3336, -1.2797, 1],
    [0.24394, 1.4733, -1.4192, -0.58535, 1],
    [-1.5322, -5.0966, 6.6779, 0.17498, 1],
    [-4.0025, -13.4979, 17.6772, -3.3202, 1],
    [-4.0173, -8.3123, 12.4547, -1.4375, 1]
]

testing_data = [
    [2.888, 0.44696, 4.5907, -0.24398, 0],
    [0.49665, 5.527, 1.7785, -0.47156, 0],
    [4.2586, 11.2962, -4.0943, -4.3457, 0],
    [1.7939, -1.1174, 1.5454, -0.26079, 0],
    [5.4021, 3.1039, -1.1536, 1.5651, 0],
    [2.5367, 2.599, 2.0938, 0.20085, 0],
    [4.6054, -4.0765, 2.7587, 0.31981, 0],
    [-1.979, 3.2301, -1.3575, -2.5819, 1],
    [-0.4294, -0.14693, 0.044265, -0.15605, 1],
    [-2.234, -7.0314, 7.4936, 0.61334, 1],
    [-4.211, -12.4736, 14.9704, -1.3884, 1],
    [-3.8073, -8.0971, 10.1772, 0.65084, 1],
    [-2.5912, -0.10554, 1.2798, 1.0414, 1],
    [-2.2482, 3.0915, -2.3969, -2.6711, 1]
]

def normalization(training_data, testing_data):
    new_training_data = deepcopy(training_data)
    new_testing_data = deepcopy(training_data)

    for j in range(0, len(new_training_data[0])-1):
        column = [row[j] for row in new_training_data]
        _min = min(column)
        _max = max(column)

        for i in range (0, len(new_training_data)):
            new_training_data[i][j] = (new_training_data[i][j] - _min) / (_max-_min)
        for i in range (0, len(new_testing_data)):
            new_testing_data[i][j] = (new_testing_data[i][j] - _min) / (_max - _min)
    return new_training_data, new_testing_data

rawDataNetwork = initialize_network(4,3,2)
normalizedDataNetworkd = initialize_network(4,3,2)

newData = normalization(training_data, testing_data)
testNormalized = newData[0]
normalizedTrainingData = newData[1]

train_network(rawDataNetwork, training_data, 0.3, 50, 2)
train_network(normalizedDataNetworkd, testNormalized, 0.3, 50, 2)

networkRaw, networkNormalize = 0, 0
for test in testing_data:
    if predict(rawDataNetwork, test) == test[-1]:
        networkRaw+=1
for test in normalizedTrainingData:
    if predict(normalizedDataNetworkd, test) == test[-1]:
        networkNormalize+=1
for row in normalizedTrainingData:
    print(row)


print("Prva mreza" + networkRaw + "Vtora mreza" + networkNormalize)