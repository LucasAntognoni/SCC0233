import numpy as np

def check(net, epsilon):
    if net > epsilon:
        return 1
    return 0


def train(X, Y, eta, threshold):

    weights = np.random.uniform(-0.5,0.5,[X.shape[1] + 1])

    # print "weights: \n", weights
    # print

    squaredError = 2 * threshold

    while squaredError > threshold:
        squaredError = 0

        for i in range(X.shape[0]):
            example = X[i,:]

            # print "example: \n", example
            # print

            e = Y[i]

            # print "e: \n", e
            # print

            example = np.append(example, [1], axis=0)

            # print "example: \n", example
            # print
            #
            # print example.shape, weights.shape
            # print

            y = check(np.dot(example, weights), 0.5)

            E = e - y

            squaredError = squaredError + E**2

            dE2dweights = 2 * E * -(example)
            weights = weights - eta * dE2dweights

        squaredError = squaredError / X.shape[0]
        print "Squared Error: ", squaredError

    return weights


def run(X, Y, weights):

    print "Expected | Obtained \n"

    for i in range(X.shape[0]):

        example = X[i,:]
        e = Y[i]

        example = np.append(example, [1], axis=0)

        net = np.dot(example, weights)
        y = check(net, 0.5)

        print e, "\t", y, "\n"


def test(eta, threshold):
    dataset = np.loadtxt('and.dat', skiprows=1)

    # print dataset
    # print

    classID = dataset.shape[1]
    # print "classID: ", classID
    # print

    Y = dataset[:,classID - 1]
    # print "Y: \n", Y
    # print

    X = dataset[:,0:classID - 1]
    # print "X: \n", X
    # print

    weights = train(X, Y, eta, threshold)
    run(X, Y, weights)

    return weights

if __name__ == '__main__':
    test(0.1, 1e-2)
