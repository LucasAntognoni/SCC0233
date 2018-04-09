import numpy as np
import csv

class Architechture():
    def __init__(self, Input, Output, Samples, Alfa):

        self.input = Input
        self.output = Output

        # Retrieved from:
        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        self.hidden = np.round(Samples / ((Input + Output) / Alfa))

        self.hidden_weights = np.random.uniform(-0.5,0.5,[self.hidden, Input])
        self.output_weights = np.random.uniform(-0.5,0.5,[Output, self.hidden])

        self.hidden_bias = np.random.uniform(-0.5,0.5,[self.hidden, 1])
        self.output_bias = np.random.uniform(-0.5,0.5,[Output, 1])

class Foward():
    def __init__(self, f_hidden, f_output, df_hidden, df_output):

        self.f_hidden = f_hidden
    	self.f_output = f_output
    	self.df_hidden = df_hidden
    	self.df_output = df_output

def converter(Y, operation):

    if operation == 0:

        classes = np.zeros((Y.shape[0], 10))

        for i in range(len(Y)):
            classes[i, int(Y[i])] = 1

        return classes
    else:
        classes = np.argmax(Y, axis=1)
        return classes

def sigmoid(net):
    return (1.0 / (1.0 + np.exp(-net)))

def sigmoid_gradient(net):
    return (sigmoid(net) * (1.0 - sigmoid(net)))

def foward(model, x):

    f_hidden = np.zeros(model.hidden)
    df_hidden = np.zeros(model.hidden)

    for j in range(model.hidden):
        net = np.dot(x, model.hidden_weights[j]) + model.hidden_bias[j]
        f_hidden[j] = sigmoid(net)
        df_hidden[j] = sigmoid_gradient(net)

    f_output = np.zeros(model.output)
    df_output = np.zeros(model.output)

    for k in range(model.output):
        net = np.dot(f_hidden, model.output_weights[k]) + model.output_bias[k]
        f_output[k] = sigmoid(net)
        df_output[k] = sigmoid_gradient(net)

    return Foward(f_hidden, f_output, df_hidden, df_output)

def backpropagation(X, Y, model, eta, threshold):

    squaredError = 2 * threshold

    while (squaredError > threshold):

        squaredError = 0

        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            fwd = foward(model, x)

            delta = y - fwd.f_output

            squaredError = squaredError + np.sum(delta * delta)

            delta_output = np.asmatrix(np.multiply(delta, fwd.df_output))
            delta_hidden = np.multiply(delta_output * model.output_weights, fwd.df_hidden)

            model.output_weights = np.asarray(model.output_weights + (eta * np.asmatrix((np.transpose(delta_output)) * np.asmatrix(fwd.f_hidden))))

            model.output_bias = np.asarray(model.output_bias + (eta * np.asmatrix(np.transpose(delta_output))))

            model.hidden_weights = np.asarray(model.hidden_weights + (eta * (np.transpose(np.asmatrix(delta_hidden)) * np.asmatrix(x))))

            model.hidden_bias = np.asarray(model.hidden_bias + (eta * np.transpose(np.asmatrix(delta_hidden))))

        squaredError = squaredError / len(X)
        print "Avarage squared error: ", squaredError

    return model

def mnist_train(eta, threshold):

    dataset = np.loadtxt('train.csv', delimiter=',', skiprows=1)

    X = np.round(dataset[0:42000,1:785] / 255)
    Y = dataset[0:42000, 0]

    Y_c = converter(Y,0)

    mlp = Architechture(X.shape[1], 10, 42000, 2)
    trained_mlp = backpropagation(X, Y_c, mlp, eta, threshold)

    # Y_t = converter(Y_c, 1)

    return trained_mlp


def mnis_test(model):

    dataset = np.loadtxt('train.csv', delimiter=',', skiprows=1)

    X = np.round(dataset[0:21000,1:785] / 255)
    Y = dataset[0:21000, 0]

    right = 0

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        fwd = foward(model, x)

        # print "Expected: ", y
        # print "Result: ", fwd.f_output[np.argmax(fwd.f_output)]
        # print "Index: ", np.argmax(fwd.f_output)
        # print

        if np.argmax(fwd.f_output) == y:
            right += 1

    print "Accuracy(%): ", (right * 100) / len(X)


def mnist_predict(model):

    dataset = np.loadtxt('test.csv', delimiter=',', skiprows=1)

    X = np.round(dataset[0:28000,0:784] / 255)

    with open('output.csv', 'wb') as myfile:

        wr = csv.writer(myfile, delimiter=',')

        wr.writerow(["ImageID", "Label"])

        for i in range(len(X)):

            x = X[i]

            fwd = foward(model, x)

            wr.writerow([i + 1, np.argmax(fwd.f_output)])

    myfile.close


if __name__ == '__main__':

    # eta = 0.1 and threshold = 1e-2
    input = (sys.argv)
    mlp = mnist_train(float(input[1]), float(input[2]))
    mnis_test(mlp)
    mnist_predict(mlp)
