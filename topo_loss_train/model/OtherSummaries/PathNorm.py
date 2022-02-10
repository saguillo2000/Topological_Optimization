import tensorflow as tf


def path_norm(model: tf.keras.Model, p=2):
    input_shape = model.input_shape
    if input_shape[0] in [0, None]:
        input_shape = (1,)+input_shape[1:]
    ones = tf.ones(input_shape)

    model = tf.keras.models.clone_model(model)

    weights = model.get_weights()
    for layer in model.layers:
        if isinstance(layer,tf.python.keras.layers.convolutional.Conv2D):
            weights, biases = layer.get_weights()
            layer.set_weights([tf.pow(tf.abs(weights),p),tf.zeros(biases.shape)])
        elif isinstance(layer,tf.python.keras.layers.core.Dense):
            weights, biases = layer.get_weights()
            layer.set_weights([tf.pow(tf.abs(weights),p),tf.zeros(biases.shape)])
        elif isinstance(layer,tf.python.keras.layers.pooling.MaxPooling2D)\
             or isinstance(layer,tf.python.keras.layers.core.Flatten):
            pass
        else:
            raise Exception("Unaccounted for layer type in PathNorm.path_norm")

    return tf.pow(tf.reduce_sum(model(ones)), 1/p).numpy()
