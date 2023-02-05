import tensorflow as tf
from tensorflow.python import keras


class KMaxPooling(keras.layers.Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(KMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.transpose(inputs, (0,3,2,1))
        x = tf.nn.top_k(x, k=self.k)[0]
        x = tf.transpose(x, (0,3,2,1))
        return x

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[1] = self.k
        return output_shape
    
    def get_config(self):
        config = {'k': self.k}
        base_config = super().get_config()
        base_config.update(config)
        return base_config