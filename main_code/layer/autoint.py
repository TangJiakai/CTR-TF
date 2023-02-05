import tensorflow as tf
from tensorflow.python import keras


class AutoIntLayer(keras.layers.Layer):
    def __init__(self, head_num, attn_dim, **kwargs):
        self.head_num = head_num
        self.attn_dim = attn_dim
        super(AutoIntLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]

        self.W_q = self.add_weight(
            name='W_Query',
            shape=(dim, self.head_num * self.attn_dim),
            trainable=True
        )        

        self.W_k = self.add_weight(
            name='W_Key',
            shape=(dim, self.head_num * self.attn_dim),
            trainable=True
        )        

        self.W_v = self.add_weight(
            name='W_Value',
            shape=(dim, self.head_num * self.attn_dim),
            trainable=True
        )       

        self.W_res = self.add_weight(
            name='W_res',
            shape=(dim, self.head_num * self.attn_dim),
            trainable=True
        ) 

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        q = tf.einsum('bij,jk', inputs, self.W_q)
        k = tf.einsum('bij,jk', inputs, self.W_k)
        v = tf.einsum('bij,jk', inputs, self.W_v)
        x = tf.einsum('bij,jk', inputs, self.W_res)

        q = tf.stack(tf.split(q, self.head_num * [self.attn_dim], axis=2), axis=0)
        k = tf.stack(tf.split(k, self.head_num * [self.attn_dim], axis=2), axis=0)
        v = tf.stack(tf.split(v, self.head_num * [self.attn_dim], axis=2), axis=0)

        weights = tf.matmul(q, k, transpose_b=True)
        weights = keras.layers.Softmax()(weights / self.attn_dim ** 0.5)

        output = tf.matmul(weights, v)
        output = tf.reshape(tf.transpose(output, perm=(1,2,0,3)), shape=[tf.shape(inputs)[0], tf.shape(inputs)[1], -1])
        output = keras.layers.ReLU()(output + x)

        return output

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.head_num * self.attn_dim)
    
    def get_config(self):
        config = {'head_num': self.head_num, 'attn_dim': self.attn_dim}
        base_config = super().get_config()
        base_config.update(config)
        return base_config