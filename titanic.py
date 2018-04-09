import numpy as np
import csv
import sys

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

#       0       1       2       3   4   5   6       7   8       9   10      11
# PassengerId Survived Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked
#             0/1      3           0/1  80  8       6

# Attributes to be used:
# Pclass: class / 3
# Sex: 0 (male), 1 (female)
# Age: age or mean_age / 80
# SibSp: value / 8
# Parch: value / 6

# CSV file was pre-processed removing not key attributes
# "Male" was changed to value 0 and "female" to 1
# Empty Age cells were filled with the mean value of all ages
# The chosen attributes were the ones that had better accuracy,
# as stated in Kaggle's Titanic discussion boards

def process_data(data):

    data_n = np.zeros((data.shape[0], 6))

    mean_age = np.median(data[:,2])
    # print mean_age
    max_age = np.amax(data[:, 2])
    # print max_age
    max_sibsp = np.amax(data[:, 3])
    # print max_sibsp
    max_parch = np.amax(data[:, 4])
    # print max_parch

    for n in range(len(data)):

        # Pclass
        data_n[n, 0] = data[n,0] / 3

        # Sex
        data_n[n, 1] = data[n, 0]

        # Age
        if(data[n, 2] == 0):
            data_n[n, 2] = mean_age / max_age
        else:
            data_n[n, 2] = data[n, 2]/ max_age

        #SibSp
        data_n[n, 3] = data[n, 3] / max_sibsp

        #Parch
        data_n[n, 4] = data[n, 4] / max_parch

        # #Embarked
        # data_n[n, 5] = data[n, 5]

    return data_n

def titanic_train(eta, threshold):

    dataset = np.loadtxt('train.csv', delimiter=',', skiprows=1)

    # dataset.shape = (891, 7)

    X = dataset[:, 2:7]
    Y = dataset[:, 1]

    X_p = process_data(X)

    # Input, Output, Samples, Alfa

    mlp = Architechture(6, 1, 891, 2)
    trained_mlp = backpropagation(X_p, Y, mlp, eta, threshold)

    return trained_mlp


def titanic_test(model):

    dataset = np.loadtxt('train.csv', delimiter=',', skiprows=1)

    X = dataset[0:420, 2:7]
    Y = dataset[0:420, 1]

    right = 0

    X_p = process_data(X)

    for i in range(len(X)):
        x = X_p[i]
        y = Y[i]

        fwd = foward(model, x)

        print int(np.round(fwd.f_output[0]))

        if np.round(int(fwd.f_output[0])) == int(y):
            right += 1

    print "Accuracy(%): ", (right * 100) / len(X)

def titanic_predict(model):

    dataset = np.loadtxt('test.csv', delimiter=',', skiprows=1)

    X = dataset[:, 1:7]
    Y = dataset[:, 0]

    X_p = process_data(X)

    with open('output.csv', 'wb') as myfile:

        wr = csv.writer(myfile, delimiter=',')

        wr.writerow(["PassengerId","Survived"])

        for i in range(len(X)):

            x = X_p[i]

            fwd = foward(model, x)

            wr.writerow([int(np.round(Y[i])), int(np.round(fwd.f_output[0]))])

    myfile.close

if __name__ == '__main__':

    input = (sys.argv)
    mlp = titanic_train(float(input[1]), float(input[2]))
    titanic_test(mlp)
    titanic_predict(mlp)
