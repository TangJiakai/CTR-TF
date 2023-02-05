import tensorflow as tf
from tensorflow.python import keras


class CIN(keras.layers.Layer):
    def __init__(self, layers, split, activation='relu', l2_reg=0., **kwargs):
        self.layers = layers
        self.split = split
        self.activation = activation
        self.l2_reg = l2_reg
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.field_num = [input_shape[1]]
        self.convs = []

        for i, size in enumerate(self.layers):
            self.convs.append(keras.layers.Conv1D(
                filters=size, 
                kernel_size=1, 
                strides=1, 
                padding='valid', 
                kernel_regularizer=keras.regularizers.l2(self.l2_reg))
            )
            if self.split:
                self.field_num.append(self.layers[i] // 2)
            else:
                self.field_num.append(self.layers[i])

        self.activations = [keras.layers.Activation(self.activation) for i in range(len(self.layers))]

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        field_num0 = inputs.shape[1]
        feature_maps = [inputs]
        outputs = []

        x0 = tf.split(inputs, [1]*dim, axis=2)
        for i in range(len(self.convs)):
            conv_layer = self.convs[i]
            x = tf.split(feature_maps[i], [1]*dim, axis=2)
            output = tf.matmul(x, x0, transpose_b=True) # (D, B, H_{k-1}, M)
            output = tf.reshape(output, shape=[dim, -1, field_num0*self.field_num[i]])
            conv_input = tf.transpose(output, perm=(1,0,2)) # (B, D, H_{k-1}*M)
            conv_output = conv_layer(conv_input) # (B, D, H_{k})
            output = tf.transpose(conv_output, perm=(0,2,1)) # (B, H_{k}, D)
            output = self.activations[i](output)
            if i == len(self.convs)-1 or not self.split:
                outputs.append(output)
                feature_maps.append(output)
            else:
                output, feature_map = tf.split(output, 2 * [self.field_num[i] // 2], axis=1)
                outputs.append(output)
                feature_maps.append(feature_map)
        
        pooling_input = tf.concat(outputs, axis=1)
        final_output = tf.reduce_sum(pooling_input, axis=2)

        return final_output

    def compute_output_shape(self, input_shape):
        output_dim = tf.reduce_sum(self.field_num[1:])
        return (None, output_dim)
    
    def get_config(self):
        config = {'layers': self.layers, 'split': self.split, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config