import tensorflow as tf
from tensorflow.python import keras


class BiLinearPooling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BiLinearPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        output = 0.5 * (tf.square(tf.reduce_sum(x, axis=1)) - tf.reduce_sum(tf.square(x), axis=1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, input_shape[2])