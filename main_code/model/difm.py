import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, DNN, PredictLayer, FM, MultiHeadAttention
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def DIFM(linear_feat_columns, dnn_feat_columns, attn_head, attn_emb_size, linear_l2_reg=0., emb_l2_reg=0., 
    dnn_hidden_units=(64,32), dnn_activation='relu', dnn_l2_reg=0., dnn_use_bn=False, dnn_dropout_rate=0., task='binary'):

    input_feats = build_input_feats(linear_feat_columns + dnn_feat_columns)

    dnn_emb_list, _ = get_input_from_feat_columns(input_feats, dnn_feat_columns, emb_l2_reg, prefix='difm')
    sparse_feat_num = len(dnn_emb_list)
    dnn_emb_input = tf.concat(dnn_emb_list, axis=1)

    Ovec = MultiHeadAttention(attn_head, attn_emb_size=attn_emb_size, use_res=True, l2_reg=dnn_l2_reg)(dnn_emb_input)
    m_vec = keras.layers.Dense(sparse_feat_num, use_bias=False)(keras.layers.Flatten()(Ovec))

    dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(keras.layers.Flatten()(dnn_emb_input))
    m_bit = keras.layers.Dense(units=sparse_feat_num, use_bias=False)(dnn_output)

    m = m_vec + m_bit

    # linear part
    linear_feat_columns = copy(linear_feat_columns)
    for i in range(len(linear_feat_columns)):
        feat = linear_feat_columns[i]
        if isinstance(feat, SparseFeat):
            linear_feat_columns[i] = linear_feat_columns[i]._replace(emb_size=1, emb_initializer=keras.initializers.Zeros())

    linear_emb_list, linear_dense_list = get_input_from_feat_columns(input_feats, linear_feat_columns, linear_l2_reg, prefix='linear')
    linear_sparse_input = tf.concat(linear_emb_list, axis=-1)
    linear_dense_input = tf.concat(linear_dense_list, axis=-1)

    linear_sparse_input = keras.layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))([linear_sparse_input, m])
    linear_logit = tf.reduce_sum(linear_sparse_input)
    linear_logit += Linear(False, linear_l2_reg)(linear_dense_input)

    # fm part
    fm_input = keras.layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))([dnn_emb_input, m])
    fm_logit = FM()(fm_input)

    # prediction part
    logit = linear_logit + fm_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model
