import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, DNN, PredictLayer, InnerProduct, FGCNNLayer
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def FGCNN(linear_feat_columns, dnn_feat_columns, filter_list, kernel_width_list, pooling_list, new_map_list, D, linear_l2_reg=0., 
    dnn_hidden_units=(64,32), dnn_activation='relu', dnn_l2_reg=0., dnn_use_bn=False, dnn_dropout_rate=0., task='binary'):
    
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
    
    # dnn part
    dnn_emb_list, _ = get_input_from_feat_columns(input_feats, dnn_feat_columns, dnn_l2_reg, prefix='dnn')
    dnn_emb_input = tf.concat(dnn_emb_list, axis=1)
    
    fg_emb_list, _ = get_input_from_feat_columns(input_feats, dnn_feat_columns, dnn_l2_reg, prefix='fg')
    fg_emb_input = tf.concat(fg_emb_list, axis=1)

    fg_output = FGCNNLayer(filter_list, kernel_width_list, pooling_list, new_map_list, dnn_l2_reg)(fg_emb_input)
    ipnn_emb_input = tf.concat([dnn_emb_input, fg_output], axis=1)

    ipnn_output = InnerProduct(D, dnn_l2_reg)(ipnn_emb_input)
    dnn_input = tf.concat([ipnn_output, keras.layers.Flatten()(ipnn_emb_input)], axis=1)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(dnn_input)
    dnn_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    # prediction part
    logit = linear_logit + dnn_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model

    
