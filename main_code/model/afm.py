import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, AFMLayer, PredictLayer
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def AFM(linear_feat_columns, dnn_feat_columns, task, attn_dim, dropout_rate, linear_l2_reg=0.0, attn_l2_reg=0.0):
    
    input_feats = build_input_feats(linear_feat_columns + dnn_feat_columns)

    # linear part
    linear_feat_columns = copy(linear_feat_columns)
    for i in range(len(linear_feat_columns)):
        feat = linear_feat_columns[i]
        if isinstance(feat, SparseFeat):
            linear_feat_columns[i] = feat._replace(emb_size=1, emb_initializer=keras.initializers.Zeros())
    
    linear_emb_list, linear_dense_list = get_input_from_feat_columns(input_feats, linear_feat_columns, linear_l2_reg, prefix='linear')
    linear_sparse_input = tf.concat(linear_emb_list, axis=-1)
    linear_dense_input = tf.concat(linear_dense_list, axis=-1)

    linear_logit = tf.reduce_sum(linear_sparse_input)
    linear_logit += Linear(False, linear_l2_reg)(linear_dense_input)

    # attention part
    attn_emb_list, _ = get_input_from_feat_columns(input_feats, dnn_feat_columns, attn_l2_reg, prefix='attention')
    
    attn_logit = AFMLayer(attn_dim, dropout_rate, attn_l2_reg)(attn_emb_list)

    logit = linear_logit + attn_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)
    return model