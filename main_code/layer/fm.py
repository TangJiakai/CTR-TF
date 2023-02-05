import tensorflow as tf
from tensorflow import keras

class FM(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input shape is expected to 3 dimensions!')
            
        return super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if len(tf.shape(inputs)) != 3:
            raise ValueError('Input shape is expected to 3 dimensions!')
        
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)

        cross_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)

        return cross_term
    
    def compute_output_shape(self, input_shape):
        return (None, 1)