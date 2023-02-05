import tensorflow as tf
from tensorflow.python import keras


class EDCNCrossNet(keras.layers.Layer):
    def __init__(self, l2_reg=0., **kwargs):
        self.l2_reg = l2_reg
        super(EDCNCrossNet, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[0][-1]

        self.V = self.add_weight(shape=(dim, 1), regularizer=keras.regularizers.l2(self.l2_reg))
        self.bias = self.add_weight(shape=(dim), initializer=keras.initializers.Zeros())
        
        super().build(input_shape)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        x_0, x_i = inputs
        x_dot = tf.tensordot(x_i, self.V, axes=[1,0])
        output = x_0 * x_dot + self.bias + x_i

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class RegulationModule(keras.layers.Layer):
    def __init__(self, tau, **kwargs):
        self.tau = tau
        super(RegulationModule, self).__init__(**kwargs)

    def build(self, input_shape):
        self.field_num, self.feat_num = input_shape[1], input_shape[2]

        self.G = self.add_weight(shape=(self.field_num, 1), initializer=keras.initializers.Ones())
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        G = keras.layers.Softmax()(self.G / self.tau)
        output = inputs * G
        output = keras.layers.Reshape([self.field_num * self.feat_num])(output)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.field_num * self.feat_num)
    
    def get_config(self):
        config = {'tau': self.tau}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class BridgeModule(keras.layers.Layer):
    def __init__(self, bridge_type, l2_reg=0., **kwargs):
        self.bridge_type = bridge_type
        self.l2_reg = l2_reg
        super(BridgeModule, self).__init__(**kwargs)

    def build(self, input_shape):
        self.feat_dim = input_shape[0][-1]
        if self.bridge_type == 'concatenation':
            self.dense = keras.layers.Dense(self.feat_dim, kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        elif self.bridge_type == 'attention_pooling':
            self.dense_x = keras.Sequential()
            self.dense_x.add(keras.layers.Dense(self.feat_dim, kernel_regularizer=keras.regularizers.l2(self.l2_reg)))
            self.dense_x.add(keras.layers.ReLU())
            self.dense_x.add(keras.layers.Dense(self.feat_dim, kernel_regularizer=keras.regularizers.l2(self.l2_reg), use_bias=False))
            self.dense_x.add(keras.layers.Softmax()) 

            self.dense_h = keras.Sequential()
            self.dense_h.add(keras.layers.Dense(self.feat_dim, kernel_regularizer=keras.regularizers.l2(self.l2_reg)))
            self.dense_h.add(keras.layers.ReLU())
            self.dense_h.add(keras.layers.Dense(self.feat_dim, kernel_regularizer=keras.regularizers.l2(self.l2_reg), use_bias=False))
            self.dense_h.add(keras.layers.Softmax())            
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x, h = inputs
        if self.bridge_type == 'add':
            output = x + h
        elif self.bridge_type == 'hadamard':
            output = x * h
        elif self.bridge_type == 'concatenation':
            output = self.dense(tf.concat([x,h], axis=-1))
        elif self.bridge_type == 'attention_pooling':
            output = self.dense_x(x) + self.dense_h(h)

        return output 

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'bridge_type': self.bridge_type, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config