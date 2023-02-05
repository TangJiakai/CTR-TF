import tensorflow as tf
from tensorflow.python import keras
from itertools import combinations


class AFMLayer(keras.layers.Layer):
    def __init__(self, attn_dim, dropout_rate, l2_reg=0., **kwargs):
        self.attn_dim = attn_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        feat_dim = input_shape[0][-1]
        
        self.W = self.add_weight(
            name='AFM_W',
            shape=(feat_dim, self.attn_dim),
            regularizer=keras.regularizers.l2(self.l2_reg),
            trainable=True
        )

        self.H = self.add_weight(
            name='AFM_H',
            shape=(self.attn_dim, 1),
            regularizer=keras.regularizers.l2(self.l2_reg),
            trainable=True
        )

        self.bias = self.add_weight(
            name='AFM_bias',
            shape=(self.attn_dim),
            initializer=keras.initializers.Zeros(),
            trainable=True
        )

        self.P = self.add_weight(
            name='AFM_P',
            shape=(feat_dim, 1),
            regularizer=keras.regularizers.l2(self.l2_reg),
            trainable=True
        )

        self.dropout = keras.layers.Dropout(self.dropout_rate)
        
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        products = []
        for x, y in combinations(inputs, 2):
            products.append(x * y)
        products = tf.concat(products, axis=1)
        wx = tf.einsum('bij,jk->bik', products, self.W)
        hidden = tf.nn.bias_add(wx, self.bias)
        hidden = tf.nn.relu(hidden)
        scores = tf.einsum('bij,jk->bik', hidden, self.H)
        scores = keras.layers.Softmax()(scores)
        attention_output = tf.multiply(products, scores)
        attention_output = tf.reduce_sum(attention_output, axis=1)
        attention_output = self.dropout(attention_output, training=training)
        output = keras.layers.dot([attention_output, self.P], axes=[-1, 0])

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)
    
    def get_config(self):
        config = {'attn_dim': self.attn_dim, 'dropout_rate': self.dropout_rate, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config