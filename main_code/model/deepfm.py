import tensorflow as tf
from tensorflow.python import keras
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from main_code.layer import Linear, FM, DNN, PredictLayer
# from itertools import chain
from copy import copy


def DeepFM(linear_feat_columns, dnn_feat_columns, task, linear_l2_reg=0.0, emb_l2_reg=0.0, dnn_hidden_units=(32,16),
    dnn_activation='relu', dnn_l2_reg=0., dnn_use_bn=False, dnn_dropout_rate=0.):

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

    # 2-order interaction part
    dnn_emb_list, dnn_dense_list = get_input_from_feat_columns(input_feats, dnn_feat_columns, emb_l2_reg, prefix='fm')
    fm_logit = FM()(tf.concat(dnn_emb_list, axis=1))

    # dnn part
    dnn_emb_input = keras.layers.Flatten()(tf.concat(dnn_emb_list, axis=-1))
    dnn_dense_input = tf.concat(dnn_dense_list, axis=-1)
    dnn_input = tf.concat([dnn_emb_input, dnn_dense_input], axis=-1)
    dnn_output = DNN(
        hidden_units=dnn_hidden_units,
        activation=dnn_activation,
        l2_reg=dnn_l2_reg,
        dropout_rate=dnn_dropout_rate,
        use_bn=dnn_use_bn
    )(dnn_input)

    dnn_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    # prediction part
    logit = linear_logit + fm_logit + dnn_logit
    outputs = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=outputs)

    return model