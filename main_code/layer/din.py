import tensorflow as tf
from tensorflow.python import keras
from main_code.layer.core import Dice


class ActivationUnit(keras.layers.Layer):
    def __init__(self, attn_hidden_units, l2_reg, **kwargs):
        self.attn_hidden_units= attn_hidden_units
        self.l2_reg = l2_reg
        super(ActivationUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dnn_layers = []
        self.activation_layers = []
        for i, attn_hidden_unit in enumerate(self.attn_hidden_units):
            self.dnn_layers.append(keras.layers.Dense(
                name='AU_' + str(i),
                units=attn_hidden_unit, 
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)))
            self.activation_layers.append(Dice())
        self.output_layer = keras.layers.Dense(1)
        super(ActivationUnit, self).build(input_shape)

    # @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None, **kwargs):
        query, key = inputs
        query = tf.repeat(query, repeats=tf.shape(key)[1], axis=1)
        x = tf.concat([query, key, query - key, query * key], axis=-1)
        for i in range(len(self.attn_hidden_units)):
            x = self.dnn_layers[i](x)
            x = self.activation_layers[i](x, training=training)
        x = self.output_layer(x)
        return x

    def compute_output_shape(self, input_shape):
        return (None, tf.shape(input_shape[1])[1], 1)

    def get_config(self):
        config = {'attn_hidden_units': self.attn_hidden_units, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class DINAttentionLayer(keras.layers.Layer):
    def __init__(self, attn_hidden_units, l2_reg=0., **kwargs):
        self.attn_hidden_units = attn_hidden_units
        self.l2_reg = l2_reg
        super(DINAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.activation_unit = ActivationUnit(self.attn_hidden_units, self.l2_reg)
        
        super().build(input_shape)

    # @tf.autograph.experimental.do_not_convert
    def call(self, inputs, mask=None, training=None, **kwargs):
        query, key = inputs
        attn_scores = self.activation_unit([query, key], training=training)
        attn_scores = tf.transpose(attn_scores, perm=(0,2,1))
        padding = tf.ones_like(attn_scores) * (-2e32 + 1)
        attn_scores = tf.where(mask, attn_scores, padding)
        attn_scores = tf.nn.softmax(attn_scores)
        output = tf.matmul(attn_scores, key)

        return output

    # @tf.autograph.experimental.do_not_convert
    def compute_output_shape(self, input_shape):
        print('='*15)
        return input_shape[0]

    def get_config(self):
        config = {'attn_hidden_units': self.attn_hidden_units, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config