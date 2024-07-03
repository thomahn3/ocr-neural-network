import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import gc
import time


def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def forward_prop(X, params):
    A = X
    caches = []
    L = len(params) // 2
    for l in range(1, L + 1):
        A_prev = A
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache)
        caches.append(cache)
    return A, caches

def cost_function(A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), 1 - Y.T))
    cost = np.squeeze(cost)
    return cost

def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def one_layer_backward(dA, cache):
    linear_cache, activation_cache = cache
    Z = activation_cache
    dZ = sigmoid_backward(dA, Z)
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = one_layer_backward(dAL, current_cache)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(grads["dA" + str(l + 2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def train(X, Y, layer_dims, epochs, lr, batch_size):
    params = init_params(layer_dims)
    cost_history = []
    m = X.shape[1]

    for i in range(epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[:, j:j+batch_size]

            Y_hat, caches = forward_prop(X_batch, params)
            cost = cost_function(Y_hat, Y_batch)
            grads = backprop(Y_hat, Y_batch, caches)
            updated_params = update_parameters(params, grads, lr)
            params = updated_params
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Cost after epoch {i}: {cost}")

        gc.collect()  # Explicitly call garbage collection to free memory

    return params, cost_history

def predict(X, params):
    Y_hat, _ = forward_prop(X, params)
    predictions = (Y_hat > 0.5).astype(int)
    return predictions

def load_and_preprocess_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, 1:].values.astype(np.float64)
    y = data.iloc[:, 0].values.astype(np.int64)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.T
    X_test = X_test.T

    encoder = OneHotEncoder(sparse_output=False)
    Y_train = encoder.fit_transform(y_train.reshape(-1, 1)).T
    Y_test = encoder.transform(y_test.reshape(-1, 1)).T

    return X_train, X_test, Y_train, Y_test

def main():
    start_time = time.time()
    csv_file = 'ocr/Data/mnist_train.csv'
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(csv_file)

    layer_dims = [784, 16, 16, 10]
    epochs = int(input("How many iterations?: "))
    learning_rate = 0.01
    batch_size = int(input("Data Processing Size?: "))  # Define a batch size

    params, cost_history = train(X_train, Y_train, layer_dims, epochs, learning_rate, batch_size)
    

    predictions = predict(X_test, params)
    accuracy = np.mean(np.argmax(predictions, axis=0) == np.argmax(Y_test, axis=0)) * 100
    print(f"{epochs} iterations")
    print(f"Batch size: {batch_size}")
    print(f"Accuracy on test set: {accuracy}%")
    print("--- %s mins ---" % ((time.time() - start_time)/60))

if __name__ == "__main__":
    main()
