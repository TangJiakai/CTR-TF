import tensorflow as tf
from tensorflow.python import keras


class InnerProduct(keras.layers.Layer):
    def __init__(self, D, l2_reg=0., **kwargs):
        self.D = D
        self.l2_reg = l2_reg
        super(InnerProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        field_num = input_shape[1]
        self.Theta = self.add_weight(
            name='InnerProduct_Theta',
            shape=(self.D, field_num),
            regularizer=keras.regularizers.l2(self.l2_reg),
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        for i in range(self.D):
            theta = tf.expand_dims(self.Theta[i], axis=1)
            output = tf.reduce_sum(tf.multiply(inputs, theta), axis=1)
            outputs.append(tf.reduce_sum(tf.square(output), axis=1, keepdims=True))

        L_p = tf.concat(outputs, axis=1)
        return L_p

    def compute_output_shape(self, input_shape):
        return (None, self.D)
    
    def get_config(self):
        config = {'D': self.D, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config
        

class OuterProduct(keras.layers.Layer):
    def __init__(self, D, l2_reg=0., **kwargs):
        self.D = D
        self.l2_reg = l2_reg
        super(OuterProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        feat_dim = input_shape[2]
        self.W = self.add_weight(
            name='OutterProduct_Theta',
            shape=(self.D, feat_dim, feat_dim),
            regularizer=keras.regularizers.l2(self.l2_reg),
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.reduce_sum(inputs, axis=1)
        x = tf.matmul(tf.expand_dims(x, axis=2), tf.expand_dims(x, axis=1))
        outputs = []
        for i in range(self.D):
            w = tf.expand_dims(self.W[i], axis=0)
            output = tf.reduce_sum(tf.multiply(w, x), axis=[1,2])
            outputs.append(tf.expand_dims(output, axis=1))

        L_p = tf.concat(outputs, axis=1)
        return L_p

    def compute_output_shape(self, input_shape):
        return (None, self.D)
    
    def get_config(self):
        config = {'D': self.D, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config