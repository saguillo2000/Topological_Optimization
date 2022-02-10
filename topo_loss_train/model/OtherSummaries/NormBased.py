import numpy as np
import tensorflow as tf


def _in_pairs(container):
    container = iter(container)
    try:
        while 1:
            yield next(container), next(container)
    except StopIteration:
        return


def _np_spectral_norm_conv(weight, input_shape):
    fft = np.fft.fft2(weight.numpy(), input_shape, axes=[0, 1])
    return np.max(np.linalg.norm(fft, 2, axis=(-1, -2)))


def _tf_spectral_norm_conv(weight, input_shape):
    #this does not account for the possibility of convolution kernels being larger than the images
    pads = [(0, input_shape[i] - weight.shape[i]) for i in range(2)] + [(0, 0)] * (len(weight.shape) - 2)
    weight = tf.pad(weight, pads)

    # the transpose here sets the dimensions of the actual kernel at the innermost
    # (as this is where fft2d does ft by default, and this cannot be modified)
    fft = tf.signal.fft2d(tf.transpose(tf.cast(weight, tf.complex64), perm=[2, 3, 0, 1]))
    fft = tf.transpose(fft)

    return tf.reduce_max(tf.norm(fft, 2, axis=(-1, -2))).numpy()


_spectral_norm_of_convolution = _tf_spectral_norm_conv


def _spec_init(model: tf.keras.Model):
    input_shape = model.input_shape[1:]
    *input_im_shape, input_channels = input_shape
    input_l2_bound = np.sqrt(np.prod(input_im_shape) * input_channels)
    network_depth = len(model.weights)

    kernel_lengths = []
    convolution_channels = []
    spectral_norms = []
    sq_centered_frobenius_norms = []

    init_of = {weight.ref():init_weight for weight, init_weight in zip(model.weights, model.initial_weights)}

    # note isinstance accounts for subclassing
    for layer in model.layers:
        if isinstance(layer,tf.python.keras.layers.convolutional.Conv2D):
            try: kernels,bias = layer.weights
            except ValueError: raise Exception("Convolution layer weights in unexpected format @ NormBased._spec_init")

            *kernel_shape, _, channels = kernels.shape
            _, *layer_input_shape, channels = layer.input_shape

            init_kernels = init_of[kernels.ref()]

            kernel_lengths.append(np.sqrt(np.prod(kernel_shape)))
            convolution_channels.append(channels)
            spectral_norms.append(_spectral_norm_of_convolution(kernels, layer_input_shape))
            sq_centered_frobenius_norms.append(tf.reduce_sum(tf.square(kernels - init_kernels)))

        elif isinstance(layer,tf.python.keras.layers.core.Dense):
            try: transform, bias = layer.weights
            except ValueError: raise Exception("Dense layer weights in unexpected format @ NormBased._spec_init")

            init_transform = init_of[transform.ref()]

            spectral_norms.append(tf.norm(transform,2).numpy())
            sq_centered_frobenius_norms.append(tf.reduce_sum(tf.square(transform - init_transform)))

        elif isinstance(layer,tf.python.keras.layers.pooling.MaxPooling2D)\
             or isinstance(layer,tf.python.keras.layers.core.Flatten):
            pass
        else:
            raise Exception("Unaccounted for layer type in NormBased._spec_init")

    proof_factor = 84 * input_l2_bound \
                   * np.sum(np.multiply(kernel_lengths, np.sqrt(convolution_channels))) \
                   + np.sqrt(np.log(4 * np.prod(input_im_shape) * network_depth))

    main_bit = np.prod(np.square(spectral_norms)) \
               * np.sum(np.divide(sq_centered_frobenius_norms, np.square(spectral_norms)))

    return proof_factor, main_bit


def spec_init(model: tf.keras.Model):
    proof_factor, main_bit = _spec_init(model)
    return proof_factor * main_bit


def spec_init_main(model: tf.keras.Model):
    proof_factor, main_bit = _spec_init(model)
    return main_bit


def spec_init_both(model: tf.keras.Model):
    proof_factor, main_bit = _spec_init(model)
    return proof_factor * main_bit, main_bit


def _spec_orig(model: tf.keras.Model):
    input_shape = model.input_shape[1:]
    *input_im_shape, input_channels = input_shape
    input_l2_bound = np.sqrt(np.prod(input_im_shape) * input_channels)
    network_depth = len(model.weights)

    kernel_lengths = []
    convolution_channels = []
    spectral_norms = []
    sq_frobenius_norms = []

    # note isinstance accounts for subclassing
    for layer in model.layers:
        if isinstance(layer,tf.python.keras.layers.convolutional.Conv2D):
            try: kernels,bias = layer.weights
            except ValueError: raise Exception("Convolution layer weights in unexpected format @ NormBased._spec_init")

            *kernel_shape, _, channels = kernels.shape
            _, *layer_input_shape, channels = layer.input_shape

            kernel_lengths.append(np.sqrt(np.prod(kernel_shape)))
            convolution_channels.append(channels)
            spectral_norms.append(_spectral_norm_of_convolution(kernels, layer_input_shape))
            sq_frobenius_norms.append(tf.reduce_sum(tf.square(kernels)))

        elif isinstance(layer,tf.python.keras.layers.core.Dense):
            try: transform, bias = layer.weights
            except ValueError: raise Exception("Dense layer weights in unexpected format @ NormBased._spec_init")

            spectral_norms.append(tf.norm(transform,2).numpy())
            sq_frobenius_norms.append(tf.reduce_sum(tf.square(transform)))

        elif isinstance(layer,tf.python.keras.layers.pooling.MaxPooling2D)\
             or isinstance(layer,tf.python.keras.layers.core.Flatten):
            pass
        else:
            raise Exception("Unaccounted for layer type in NormBased._spec_init")

    proof_factor = 84 * input_l2_bound \
                   * np.sum(np.multiply(kernel_lengths, np.sqrt(convolution_channels))) \
                   + np.sqrt(np.log(4 * np.prod(input_im_shape) * network_depth))

    spectral_product = np.prod(np.square(spectral_norms))
    ratio_sum = np.sum(np.divide(sq_frobenius_norms, np.square(spectral_norms)))

    return proof_factor, spectral_product, ratio_sum

def spec_orig(model: tf.keras.Model):
    proof_factor, spectral_product, ratio_sum = _spec_orig(model)
    main_bit = spectral_product * ratio_sum
    return proof_factor * main_bit


def spec_orig_main(model: tf.keras.Model):
    proof_factor, spectral_product, ratio_sum = _spec_orig(model)
    main_bit = spectral_product * ratio_sum
    return main_bit


def spec_orig_both(model: tf.keras.Model):
    proof_factor, spectral_product, ratio_sum = _spec_orig(model)
    main_bit = spectral_product * ratio_sum
    return proof_factor * main_bit, main_bit


def spec_orig_all(model: tf.keras.Model):
    proof_factor, spectral_product, ratio_sum = _spec_orig(model)
    main_bit = spectral_product * ratio_sum
    return proof_factor * main_bit, main_bit, spectral_product


def _spec_all(model: tf.keras.Model):
    input_shape = model.input_shape[1:]
    *input_im_shape, input_channels = input_shape
    input_l2_bound = np.sqrt(np.prod(input_im_shape) * input_channels)
    network_depth = len(model.weights)

    kernel_lengths = []
    convolution_channels = []
    spectral_norms = []
    sq_frobenius_norms = []
    sq_centered_frobenius_norms = []

    init_of = {weight.ref():init_weight for weight, init_weight in zip(model.weights, model.initial_weights)}

    # note isinstance accounts for subclassing
    for layer in model.layers:
        if isinstance(layer,tf.python.keras.layers.convolutional.Conv2D):
            try: kernels,bias = layer.weights
            except ValueError: raise Exception("Convolution layer weights in unexpected format @ NormBased._spec_init")

            *kernel_shape, _, channels = kernels.shape
            _, *layer_input_shape, channels = layer.input_shape

            init_kernels = init_of[kernels.ref()]

            kernel_lengths.append(np.sqrt(np.prod(kernel_shape)))
            convolution_channels.append(channels)
            spectral_norms.append(_spectral_norm_of_convolution(kernels, layer_input_shape))
            sq_frobenius_norms.append(tf.reduce_sum(tf.square(kernels)))
            sq_centered_frobenius_norms.append(tf.reduce_sum(tf.square(kernels - init_kernels)))

        elif isinstance(layer,tf.python.keras.layers.core.Dense):
            try: transform, bias = layer.weights
            except ValueError: raise Exception("Dense layer weights in unexpected format @ NormBased._spec_init")

            init_transform = init_of[transform.ref()]

            spectral_norms.append(tf.norm(transform,2).numpy())
            sq_frobenius_norms.append(tf.reduce_sum(tf.square(transform)))
            sq_centered_frobenius_norms.append(tf.reduce_sum(tf.square(transform - init_transform)))

        elif isinstance(layer,tf.python.keras.layers.pooling.MaxPooling2D)\
             or isinstance(layer,tf.python.keras.layers.core.Flatten):
            pass
        else:
            raise Exception("Unaccounted for layer type in NormBased._spec_init")

    proof_factor = 84 * input_l2_bound \
                   * np.sum(np.multiply(kernel_lengths, np.sqrt(convolution_channels))) \
                   + np.sqrt(np.log(4 * np.prod(input_im_shape) * network_depth))

    spectral_product = np.prod(np.square(spectral_norms))
    ratio_sum_init = np.sum(np.divide(sq_centered_frobenius_norms, np.square(spectral_norms)))
    ratio_sum_orig = np.sum(np.divide(sq_frobenius_norms, np.square(spectral_norms)))

    return proof_factor, spectral_product, ratio_sum_init, ratio_sum_orig

def spec_all(model: tf.keras.Model):
    proof_factor, spectral_product, ratio_sum_init, ratio_sum_orig = _spec_all(model)
    main_bit_init = spectral_product*ratio_sum_init
    main_bit_orig = spectral_product*ratio_sum_orig
    return (proof_factor * main_bit_init,
            proof_factor * main_bit_orig,
            main_bit_init,
            main_bit_orig,
            spectral_product)





def frob_sum(model: tf.keras.Model):
    """ref. [Fantastic generalization Measures and where to find them] for clarification on naming"""

    # strictly speaking, the frobenius norm is not defined for vectors,
    # and indeed the regular tf frobenius norm implementation will throw an exception if a vector is passed
    # this is why a helper function is used
    frobenius_norm = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x)))
    norms = tf.stack([frobenius_norm(weight) for weight in model.weights])

    network_depth = len(model.layers)
    return network_depth*tf.pow(tf.reduce_prod(norms),2/network_depth).numpy()
