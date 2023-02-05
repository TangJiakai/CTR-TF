import tensorflow as tf
from tensorflow.python import keras
from itertools import combinations


class SENet(keras.layers.Layer):
    def __init__(self, reduce_ratio, l2_reg, **kwargs):
        self.reduce_ratio = reduce_ratio
        self.l2_reg = l2_reg

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.field_num = len(input_shape)

        self.w1 = self.add_weight(
            name='SENet_w1',
            shape=(self.field_num // self.reduce_ratio, self.field_num),
            regularizer=keras.regularizers.l2(self.l2_reg),
            trainable=True,
        )

        self.w2 = self.add_weight(
            name='SENet_w2',
            shape=(self.field_num, self.field_num // self.reduce_ratio),
            regularizer=keras.regularizers.l2(self.l2_reg),
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = tf.concat(inputs, axis=1)
        x = tf.reduce_mean(inputs, axis=2, keepdims=True)

        A1 = tf.nn.relu(tf.einsum('ij,bjk->bik', self.w1, x))
        A2 = tf.nn.relu(tf.einsum('ij,bjk->bik', self.w2, A1))

        output = inputs * A2
        output = tf.split(output, self.field_num, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'reduce_ratio': self.reduce_ratio, 'l2_reg': self.l2_reg}
        base_config = super().get_config()        
        base_config.update(config)
        return base_config


class BiLinearInteraction(keras.layers.Layer):
    def __init__(self, bilinear_type, l2_reg, **kwargs):
        self.bilinear_type = bilinear_type
        self.l2_reg = l2_reg

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.field_num = len(input_shape)
        dim = input_shape[0][-1]
        if self.bilinear_type == 'all':
            self.W = self.add_weight(
                name='BiLinear_all',
                shape=(dim, dim),
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            )
        elif self.bilinear_type == 'each':
            self.W = [
                self.add_weight(
                    name='BiLinear_each' + str(i),
                    shape=(dim, dim),
                    regularizer=keras.regularizers.l2(self.l2_reg),
                    trainable=True
                ) for i in range(self.field_num-1)
            ]
        elif self.bilinear_type == 'interaction':
            self.W = [
                self.add_weight(
                    name='BiLinear_each' + str(i),
                    shape=(dim, dim),
                    regularizer=keras.regularizers.l2(self.l2_reg),
                    trainable=True
                ) for i in range(self.field_num * (self.field_num-1) // 2)
            ]

        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        if self.bilinear_type == 'all':
            outputs = [
                tf.matmul(inputs[i], self.W) * inputs[j]
                for i, j in combinations(range(self.field_num), 2)
            ]
        elif self.bilinear_type == 'each':
            outputs =[
                tf.matmul(inputs[i], self.W[i]) * inputs[j]
                for i, j in combinations(range(self.field_num), 2)
            ]
        elif self.bilinear_type == 'interaction':
            outputs = [
                tf.matmul(inputs[i], W) * inputs[j]
                for (i,j), W in zip(combinations(range(self.field_num), 2), self.W)
            ]
        
        output = keras.layers.Flatten()(tf.stack(outputs, axis=1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.field_num * (self.field_num-1) //2, 1)

    def get_config(self):
        config = {'bilinear_type': self.bilinear_type, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config