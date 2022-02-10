import tensorflow as tf
from tensorflow.python.keras import Sequential


def load_model(config):
    model_instance = _model_def_to_keras_sequential(config['model_config'])
    model_instance.build([0] + config['input_shape'])
    return model_instance


def _model_def_to_keras_sequential(model_def):
    """Convert a model json to a Keras Sequential model.

    Args:
        model_def: A list of dictionaries, where each dict describes a layer to add
            to the model.

    Returns:
        A Keras Sequential model with the required architecture.
    """

    def _cast_to_integer_if_possible(dct):
        dct = dict(dct)
        for k, v in dct.items():
            if isinstance(v, float) and v.is_integer():
                dct[k] = int(v)
        return dct

    def parse_layer(layer_def):
        layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
        # layer_cls = wrap_layer(layer_cls)
        kwargs = dict(layer_def)
        del kwargs['layer_name']
        return _wrap_layer(layer_cls, **_cast_to_integer_if_possible(kwargs))
        # return layer_cls(**_cast_to_integer_if_possible(kwargs))

    return Sequential([parse_layer(l) for l in model_def])


def _wrap_layer(layer_cls, *args, **kwargs):
    """Wraps a layer for computing the jacobian wrt to intermediate layers."""

    class wrapped_layer(layer_cls):
        def __call__(self, x, *args, **kwargs):
            self._last_seen_input = x
            return super(wrapped_layer, self).__call__(x, *args, **kwargs)

    return wrapped_layer(*args, **kwargs)
