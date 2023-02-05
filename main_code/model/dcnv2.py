import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, DNN, DCNv2CrossNet, PredictLayer
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def DCNv2(linear_feat_columns, dnn_feat_columns, cross_num, expert_num, low_rank, linear_l2_reg=0., emb_l2_reg=0., dcnv2_l2_reg=0., dnn_hidden_units=(64,32),
     dnn_activation='relu', dnn_l2_reg=0., dnn_use_bn=False, dnn_dropout_rate=0., task='binary'):
     
     input_feats = build_input_feats(linear_feat_columns + dnn_feat_columns)

     # linear part
     linear_feat_columns = copy(linear_feat_columns)
     for i in range(len(linear_feat_columns)):
          feat = linear_feat_columns[i]
          if isinstance(feat, SparseFeat):
               linear_feat_columns[i] = linear_feat_columns[i]._replace(emb_size=1, emb_initializer=keras.initializers.Zeros())

     linear_emb_list, linear_dense_list = get_input_from_feat_columns(input_feats, linear_feat_columns, linear_l2_reg, prefix='linear')
     linear_sparse_input = tf.concat(linear_emb_list, axis=-1)
     linear_dense_input = tf.concat(linear_dense_list, axis=-1)

     linear_logit = tf.reduce_sum(linear_sparse_input)
     linear_logit += Linear(False, linear_l2_reg)(linear_dense_input)

     # cross net & dnn 
     dnn_emb_list, dnn_dense_list = get_input_from_feat_columns(input_feats, dnn_feat_columns, emb_l2_reg, prefix='cross_and_dnn')
     dnn_emb_input = keras.layers.Flatten()(tf.concat(dnn_emb_list, axis=-1))
     dnn_dense_input = tf.concat(dnn_dense_list, axis=-1)
     dnn_input = tf.concat([dnn_emb_input, dnn_dense_input], axis=-1)

     cross_output = DCNv2CrossNet(cross_num, expert_num, low_rank, dcnv2_l2_reg)(dnn_input)

     dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(dnn_input)
     
     cross_and_dnn_output = tf.concat([cross_output, dnn_output], axis=-1)
     
     cross_and_dnn_logit = Linear(False, dcnv2_l2_reg)(cross_and_dnn_output)

     # prediction part
     logit = linear_logit + cross_and_dnn_logit
     output = PredictLayer(task)(logit)

     model = keras.Model(inputs=input_feats, outputs=output)

     return model