import tensorflow as tf
from tensorflow.python import keras


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, attn_head, attn_emb_size, use_res=True, l2_reg=0., **kwargs):
        self.attn_head = attn_head
        self.attn_emb_size = attn_emb_size
        self.use_res = use_res
        self.l2_reg = l2_reg
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.feat_dim = input_shape[-1]

        self.W_query = self.add_weight(
            name='W_query',
            shape=(self.feat_dim, self.attn_emb_size * self.attn_head),
            regularizer=keras.regularizers.l2(self.l2_reg)
        )

        self.W_key = self.add_weight(
            name='W_key',
            shape=(self.feat_dim, self.attn_emb_size * self.attn_head),
            regularizer=keras.regularizers.l2(self.l2_reg)
        )

        self.W_value = self.add_weight(
            name='W_value',
            shape=(self.feat_dim, self.attn_emb_size * self.attn_head),
            regularizer=keras.regularizers.l2(self.l2_reg)
        )

        if self.use_res:
            self.W_res = self.add_weight(
                name='W_res',
                shape=(self.feat_dim, self.attn_emb_size * self.attn_head),
                regularizer=keras.regularizers.l2(self.l2_reg)
            )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        
        Q = tf.matmul(inputs, self.W_query)
        K = tf.matmul(inputs, self.W_key)
        V = tf.matmul(inputs, self.W_value)

        Q = tf.stack(tf.split(Q, [self.attn_emb_size] * self.attn_head, axis=-1), axis=1)
        K = tf.stack(tf.split(K, [self.attn_emb_size] * self.attn_head, axis=-1), axis=1)
        V = tf.stack(tf.split(V, [self.attn_emb_size] * self.attn_head, axis=-1), axis=1)

        QK_dot = tf.matmul(Q, K, transpose_b=True)
        QK_dot = keras.layers.Softmax()(QK_dot / tf.cast(self.feat_dim, tf.float32) ** 0.5)
        output = tf.matmul(QK_dot, V)

        if self.use_res:
            res = tf.matmul(inputs, self.W_res)
            res = tf.stack(tf.split(res, [self.attn_emb_size] * self.attn_head, axis=-1), axis=1)
            output += res

        output = tf.transpose(output, perm=(0,2,1,3))
        output = tf.reshape(output, shape=tf.shape(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {'attn_head': self.attn_head, 
                'attn_emb_size': self.attn_emb_size,
                'l2_reg': self.l2_reg,
                'use_res': self.use_res}
        base_config = super().get_config()
        base_config.update(config)
        return base_config