import tensorflow as tf
from tensorflow.python import keras
from itertools import combinations


class FFMLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        field_num = inputs.shape[1]
        logit = 0
        for i, j in combinations(list(range(field_num)), 2):
            feat_i = inputs[:, i, j]
            feat_j = inputs[:, j, i]
            logit += keras.layers.dot([feat_i, feat_j], axes=1)

        return logit

    def compute_output_shape(self, input_shape):
        return (None, 1)
    
