from tensorflow import where


def LReLU(x):
    alpha = 0.01
#    return K.tf.nn.leaky_relu(x, alpha=alpha)
    return where(x>0, x, alpha * x)
