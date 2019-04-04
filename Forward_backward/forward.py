import numpy as np
from Forward_backward.activate import sigmoid

def initialize_parameters(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W"+str(l)] = np.random.rand(layers_dim[l], layers_dim[l-1])*0.01
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters["w"+str(l)].shape == (layers_dims[l], layers_dims[l-1])
        assert parameters["b" + str(l)].shape == (layers_dims[l], 1)
    return parameters

def linear_forward(A_pre, W, b):
    Z = np.dot(w,A_pre)+b
    cache = (A_pre,w,b)
    return Z, cache

def linear_activation_forward(A_pre, W, b,activation_fn):

    if activation_fn == 'sigmoid':
        Z, linear_cache = linear_forward(A_pre,W,b)
        A, activation_cache =sigmoid(Z)
        cache =(linear_cache, activation_cache)
        return A ,cache

def L_model_forward(X, parameters, hidden_layers_activation_fn='relu'):
    A=X
    caches = []
    L= len(parameters)//2
    for l in range(1, L):
        A_prev= A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)],
                                             parameters["b"+str(l)],activation_fn=hidden_layers_activation_fn
                                             )
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], activation_fn="sigmoid")
    caches.append(cache)
    return AL, caches
# compute cross-entropy cost

def compute_cost(AL, y):
    m = y.shape[1]
    cost = -(1/m)*np.sum(np.multiply(y,np.log(AL))+ np.multiply(1-y, np.log(1-AL)))
    return cost










