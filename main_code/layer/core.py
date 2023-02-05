import tensorflow as tf
from tensorflow.python import keras


class PredictLayer(keras.layers.Layer):
    def __init__(self, task, use_bias=True, **kwargs):
        if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError('task must be in binary, multiclass or regression!')
        
        self.task = task
        self.use_bias = use_bias

        super(PredictLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.global_bias = self.add_weight(
                'global_bias', 
                shape=(1,), 
                initializer=keras.initializers.Zeros(),
                trainable=True
            )
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias)
        if self.task == 'binary':
            x = tf.sigmoid(x)
        
        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))


class DNN(keras.layers.Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0., dropout_rate=0., use_bn=False, **kwargs):
        
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):

        self.fcs = []
        self.bns = []
        self.dropouts = []
        self.activations = []
        for units in self.hidden_units:
            self.fcs.append(keras.layers.Dense(
                units=units,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            ))

            if self.use_bn:
                self.bns.append(keras.layers.BatchNormalization())
            
            self.dropouts.append(keras.layers.Dropout(self.dropout_rate))
            
            self.activations.append(keras.layers.Activation(self.activation))

        super().build(input_shape)
    
    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for i in range(len(self.hidden_units)):
            x = self.fcs[i](x)
            if self.use_bn:
                x = self.bns[i](x, training=training)
            x = self.activations[i](x, training=training)
            x = self.dropouts[i](x, training=training)
        
        return x
    
    def compute_output_shape(self, input_shape):
        shape = input_shape
        if len(self.hidden_units) > 0:
            shape = (input_shape[:-1] + self.hidden_units[-1],)

        return shape
    
    def get_config(self):
        config = {'hidden_units': self.hidden_units,
                'activation': self.activation,
                'l2_reg': self.l2_reg,
                'dropout_rate': self.dropout_rate,
                'use_bn': self.use_bn}
        base_config = super().get_config()
        base_config.update(config)
        return base_config


class Dice(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = keras.layers.BatchNormalization(center=False, scale=False)
        self.alphas = self.add_weight(
            name='Dice_Alpha',
            shape=(input_shape[-1]),
            initializer=keras.initializers.Zeros()
        )
        super().build(input_shape)
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        x = self.bn(inputs, training=training)
        x_p = keras.activations.sigmoid(x)
        output = x_p * inputs + (1-x_p) * self.alphas * inputs

        return output
