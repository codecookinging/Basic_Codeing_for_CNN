from Forward_backward.activate import sigmoid,tanh,relu
import numpy as np

def sigmoid_gradient(da, z):
    a, z = sigmoid(z)
    dz = da*a(1-a)

    return dz

def tanh_gradient(da, z):
    a, z = tanh(z)
    dz= da*(1-np.square(a))
    return dz
def relu_gradient(da,z):
    a,z = relu(z)
    dz= np.multiply(da, np.int64(a>0))
    return dz

def linear_backward(dz, cache):
    a_prev, w, b =cache
    m =a_prev.shape[1]
    dw = (1/m)*np.dot(dz, a_prev.T)
    db = (1/m)*np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(w.T,dz)
    return da_prev, dw, db
def linear_activate_backward(da, cache, activattion_fn):
    linear_cache, activation_cahce =cache
    if activation_fn == 'sigmoid':
        dz = sigmoid_gradient(da, activation_cahce)
        da_prev, dw, db = linear_backward(dz, linear_cache)
    elif activattion_fn == "relu":
        dz = relu_gradient(da, activation_cahce)
        da_prev, dw, db = linear_backward(dz, linear_cache)
    return da_prev, dw, db


def L_model_backward(al, y,caches, hidden_layers_activation_fn ="relu"):
    y = y.reshape(al.shape)
    l = len(caches)
    grads = {}
    dal = np.divide(al-y, np.multiply(al, 1-al))
    grads = []
    grads["da" + str(l - 1)], grads["dw" + str(l)], grads[
        "db" + str(l)] = linear_activation_backward(
            dal, caches[l - 1], "sigmoid")
    for l in range(l-1, 0, -1):
        current_cache = caches[l-1]
        grads["da" + str(l - 1)], grads["dw" + str(l)], grads[
            "db" + str(l)] = linear_activate_backward(
                grads["da" + str(l)], current_cache,
                hidden_layers_activation_fn)
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters



