import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, AutoIntLayer, PredictLayer
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def AutoInt(linear_feat_columns, dnn_feat_columns, layer_num, head_num, attn_dim, linear_l2_reg=0., emb_l2_reg=0., task='binary'):
    input_feats = build_input_feats(linear_feat_columns + dnn_feat_columns)

    # linear part
    linear_feat_columns = copy(linear_feat_columns)
    for i in range(len(linear_feat_columns)):
        feat = linear_feat_columns[i]
        if isinstance(feat, SparseFeat):
            linear_feat_columns[i] = linear_feat_columns[i]._replace(emb_size=1, emb_initializer=keras.initializers.zeros())
    linear_emb_list, linear_dense_list = get_input_from_feat_columns(input_feats, linear_feat_columns, linear_l2_reg, prefix='linear')
    linear_sparse_input = tf.concat(linear_emb_list, axis=-1)
    linear_dense_input = tf.concat(linear_dense_list, axis=-1)

    linear_logit = tf.reduce_sum(linear_sparse_input)
    linear_logit += Linear(False, linear_l2_reg)(linear_dense_input)

    # multi-head attention
    attn_emb_list, attn_dense_list = get_input_from_feat_columns(input_feats, dnn_feat_columns, emb_l2_reg, prefix='Attention')
    attn_emb_input = tf.concat(attn_emb_list, axis=1)

    for i in range(layer_num):
        attn_emb_input = AutoIntLayer(head_num, attn_dim)(attn_emb_input)
    
    attn_output = keras.layers.Flatten()(attn_emb_input)

    attn_logit = keras.layers.Dense(1, use_bias=False)(attn_output)

    logit = linear_logit + attn_logit
    output = PredictLayer(task)(logit)
    model = keras.Model(inputs=input_feats, outputs=output)
    
    return model