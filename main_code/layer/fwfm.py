import tensorflow as tf
from tensorflow.python import keras
from itertools import combinations


class FwFMLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FwFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.field_num = len(input_shape)
        self.W = self.add_weight(
            name='FwFM_weights',
            shape=(self.field_num, self.field_num),
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        logit = 0
        for (i, j), (feat_i, feat_j) in zip(combinations(list(range(self.field_num)), 2), combinations(inputs, 2)):
            logit += self.W[i,j] * tf.squeeze(keras.layers.dot([feat_i, feat_j], axes=2), axis=2)

        return logit

    def compute_output_shape(self, input_shape):
        return input_shape