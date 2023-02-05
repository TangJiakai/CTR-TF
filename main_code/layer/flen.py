import tensorflow as tf
from tensorflow.python import keras
from itertools import combinations


class FieldWiseBiInteraction(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FieldWiseBiInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.group_num = len(input_shape)

        self.r = self.add_weight(
            name='FieldWiseBiInteraction_r',
            shape=(self.group_num, self.group_num),
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        mf_embs = []
        for x in inputs:
            emb = tf.reduce_sum(x, axis=1)
            mf_embs.append(emb)
        mf_output = 0
        for i, j in combinations(list(range(self.group_num)), 2):
            mf_output += mf_embs[i] * mf_embs[j] * self.r[i][j]
        
        fm_output = 0
        for i in range(self.group_num):
            square_of_sum = tf.square(tf.reduce_sum(inputs[i], axis=1))
            sum_of_square = tf.reduce_sum(tf.square(inputs[i]), axis=1)
            fm_output = (square_of_sum - sum_of_square) * self.r[i][i]

        return mf_output + fm_output

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][-1])