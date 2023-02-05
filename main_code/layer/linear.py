import tensorflow as tf
from tensorflow.python import keras

class Linear(keras.layers.Layer):
    def __init__(self, use_bias=False, l2_reg=0., **kwargs):
        self.use_bias = use_bias
        self.l2_reg = l2_reg

        self.linear = keras.layers.Dense(
            units=1, 
            use_bias=use_bias, 
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )

        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = self.linear(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = {'use_bias': self.use_bias, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config