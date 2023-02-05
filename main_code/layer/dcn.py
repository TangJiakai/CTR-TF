import tensorflow as tf
from tensorflow.python import keras


class DCNCrossNet(keras.layers.Layer):
    def __init__(self, cross_num, l2_reg=0., **kwargs):
        self.cross_num = cross_num
        self.l2_reg = l2_reg
        super(DCNCrossNet, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]

        self.kernels = []
        for i in range(self.cross_num):
            self.kernels.append(self.add_weight(
                name='kernel' + str(i),
                shape=(dim,1),
                initializer=keras.initializers.glorot_normal(),
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            ))

        self.bias = []
        for i in range(self.cross_num):
            self.bias.append(self.add_weight(
                name='bias'+str(i),
                shape=(dim,1),
                initializer=keras.initializers.glorot_normal(),
                trainable=True
            ))
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.cross_num):
            xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1,0))
            dot = x_0 * xl_w
            x_l = dot + self.bias[i] + x_l
        x_l = tf.squeeze(x_l, axis=2)

        return x_l

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {'layer_num': self.cross_num, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config