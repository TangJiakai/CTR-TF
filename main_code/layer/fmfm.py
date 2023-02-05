import tensorflow as tf
from tensorflow.python import keras
from itertools import combinations
from collections import defaultdict


class FmFMLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FmFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.field_num = len(input_shape)
        feat_dim = input_shape[0][-1]
        self.M = defaultdict(dict)
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                self.M[i][j] = self.add_weight(
                    name='W_' + str(i) + '_' + str(j),
                    shape=(feat_dim, feat_dim),
                    trainable=True
                )
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        logit = 0
        for (i, j), (feat_i, feat_j) in zip(combinations(list(range(self.field_num)), 2), combinations(inputs, 2)):
            feat_i = tf.matmul(feat_i, self.M[i][j])
            logit += tf.squeeze(keras.layers.dot([feat_i, feat_j], axes=2), axis=2)

        return logit

    def compute_output_shape(self, input_shape):
        return (None, 1)