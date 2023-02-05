import tensorflow as tf
from tensorflow.python import keras


class DCNv2CrossNet(keras.layers.Layer):
    def __init__(self, cross_num, expert_num, low_rank, l2_reg, **kwargs):
        self.cross_num = cross_num
        self.expert_num = expert_num
        self.low_rank = low_rank
        self.l2_reg = l2_reg

        super(DCNv2CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        
        self.Vs = []
        self.Us = []
        self.Cs = []
        self.bias = []
        for i in range(self.cross_num):
            self.Vs.append(self.add_weight(
                name='V_list' + str(i),
                shape=(self.expert_num, self.low_rank, dim),
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            ))      

            self.Us.append(self.add_weight(
                name='U_list' + str(i),
                shape=(self.expert_num, dim, self.low_rank),
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            ))          
            
            self.Cs.append(self.add_weight(
                name='C_list' + str(i),
                shape=(self.expert_num, self.low_rank, self.low_rank),
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            ))

            self.bias.append(self.add_weight(
                name='bias' + str(i),
                shape=(dim, 1),
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            ))

        self.gatings = []
        for i in range(self.expert_num):
            self.gatings.append(keras.layers.Dense(
                units=1,
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            ))

        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)
        x_l = x0
        for i in range(self.cross_num):
            gating_score_list = []
            expert_output_list = []
            for j in range(self.expert_num):
                gating_score = self.gatings[j](tf.squeeze(x_l, axis=2))
                gating_score_list.append(gating_score)

                v_x = tf.einsum('ij,bjk->bik', self.Vs[i][j], x_l)
                v_x = tf.tanh(v_x)
                v_x = tf.einsum('ij,bjk->bik', self.Cs[i][j], v_x)
                v_x = tf.tanh(v_x)

                uv_x = tf.einsum('ij,bjk->bik', self.Us[i][j], v_x)
                
                dot_ = uv_x + self.bias[i]
                dot_ = dot_ * x0

                expert_output_list.append(tf.squeeze(dot_, axis=2))
            
            gating_scores = tf.stack(gating_score_list, axis=1) # (b, expert_num, 1)
            expert_outputs = tf.stack(expert_output_list, axis=2) # (b, dim, expert_num)
            
            moe_out = tf.matmul(expert_outputs, gating_scores) # (b, dim, 1)
            x_l = moe_out + x_l
        
        x_l = tf.squeeze(x_l, axis=2)
        return x_l
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'cross_num': self.cross_num, 'expert_num': self.expert_num, 'low_rank': self.low_rank, 'l2_reg': self.l2_reg}
        base_config = super().get_config()
        base_config.update(config)
        return base_config