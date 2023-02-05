import tensorflow as tf
from tensorflow.python import keras
from main_code.inputs import build_input_feats, get_input_from_feat_columns, get_input_from_group_feat_columns
from main_code.layer import Linear, FieldWiseBiInteraction, DNN, PredictLayer
from main_code.feature import SparseFeat
from itertools import chain


def FLEN(linear_feat_columns, dnn_feat_columns, task, linear_l2_reg=0., dnn_hidden_units=(32,16),
    dnn_activation='relu', dnn_l2_reg=0., dnn_use_bn=False, dnn_dropout_rate=0.):
    input_feats = build_input_feats(linear_feat_columns + dnn_feat_columns)

    for i in range(len(linear_feat_columns)):
        feat = linear_feat_columns[i]
        if isinstance(feat, SparseFeat):
            linear_feat_columns[i] = linear_feat_columns[i]._replace(emb_size=1, emb_initializer=keras.initializers.Zeros())

    linear_emb_list, _ = get_input_from_feat_columns(input_feats, linear_feat_columns, linear_l2_reg, prefix='linear')
    linear_sparse_input = tf.squeeze(tf.concat(linear_emb_list, axis=-1), axis=1)

    # linear_logit = tf.reduce_sum(linear_sparse_input)

    dnn_group_emb_feats, _ = get_input_from_group_feat_columns(input_feats, dnn_feat_columns, dnn_l2_reg, prefix='dnn')
    dnn_group_emb_input = [tf.concat(list(group_feats.values()), axis=1) for group_feats in dnn_group_emb_feats.values()]

    fm_mf_output = FieldWiseBiInteraction()(dnn_group_emb_input)
    dnn_input = tf.concat([tf.concat(list(group_feats.values()), axis=1) for group_feats in dnn_group_emb_feats.values()], axis=1)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(keras.layers.Flatten()(dnn_input))
    dnn_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    fm_mf_linear_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(tf.concat([linear_sparse_input, fm_mf_output], axis=1))
    logit = keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(tf.concat([fm_mf_linear_logit, dnn_logit], axis=1))
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model



