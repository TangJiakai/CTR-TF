import tensorflow as tf
from tensorflow.python import keras


class FGCNNLayer(keras.layers.Layer):
    def __init__(self, filter_list, kernel_width_list, pooling_list, new_map_list, l2_reg=0., **kwargs):
        self.filter_list = filter_list
        self.kernel_width_list = kernel_width_list
        self.pooling_list = pooling_list
        self.new_map_list = new_map_list
        self.l2_reg = l2_reg
        super(FGCNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[2]
        field_num = input_shape[1]

        self.conv_layers = []
        self.pooling_layers = []
        self.dense_layers = []
        self.dense_field_num_list = []
        for i in range(len(self.filter_list)):
            self.conv_layers.append(
                keras.layers.Conv2D(
                    name='Conv_' + str(i),
                    filters=self.filter_list[i],
                    kernel_size=(self.kernel_width_list[i], 1),
                    padding='same',
                    activation=keras.activations.tanh,
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg)
                )
            )
            
            self.pooling_layers.append(keras.layers.MaxPool2D(name='Pool_'+str(i), pool_size=(self.pooling_list[i], 1)))
            
            field_num //= self.pooling_list[i]
            self.dense_field_num_list.append(field_num)
            self.dense_layers.append(keras.layers.Dense(
                name='Dense_' + str(i),
                units=(field_num * dim * self.new_map_list[i]),
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            ))
        
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feat_dim = tf.shape(inputs)[2]
        x = tf.expand_dims(inputs, axis=-1)
        
        output_list = []
        for i in range(len(self.filter_list)):
            conv_output = self.conv_layers[i](x)
            pool_output = self.pooling_layers[i](conv_output)
            dense_input = keras.layers.Flatten()(pool_output)
            dense_output = self.dense_layers[i](dense_input)
            output_list.append(tf.reshape(dense_output, shape=(-1, self.dense_field_num_list[i] * self.new_map_list[i], feat_dim)))
            x = pool_output

        output = tf.concat(output_list, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        feat_dim = input_shape[-1]
        new_feat_num = sum([field_num * new_map_num for field_num, new_map_num in zip(self.dense_field_num_list, self.new_map_list)])
        return (None, new_feat_num, feat_dim)
    
    def get_config(self):
        config = {
            'filter_list': self.filter_list, 
            'kernel_width_list': self.kernel_width_list,
            'pooling_list': self.pooling_list,
            'new_map_list': self.new_map_list,
            'l2_reg': self.l2_reg
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config