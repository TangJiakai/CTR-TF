import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, PredictLayer, DNN, CIN
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def xDeepFM(feat_columns, task, linear_l2_reg=0.0, cin_layers=(128,128), cin_split=True, cin_activation='relu',
    dnn_hidden_units=(64,32), dnn_activation='relu', dnn_l2_reg=0.0, dnn_dropout_rate=0.0, dnn_use_bn=False):

    input_feats = build_input_feats(feat_columns)

    # linear part
    linear_feat_columns = copy(feat_columns)
    for i in range(len(linear_feat_columns)):
        feat = linear_feat_columns[i]
        if isinstance(feat, SparseFeat):
            linear_feat_columns[i] = linear_feat_columns[i]._replace(emb_size=1, emb_initializer=keras.initializers.Zeros())

    linear_emb_list, linear_dense_list = get_input_from_feat_columns(input_feats, linear_feat_columns, linear_l2_reg, prefix='linear')
    linear_sparse_input = tf.concat(linear_emb_list, axis=-1)
    linear_dense_input = tf.concat(linear_dense_list, axis=-1)

    linear_logit = tf.reduce_sum(linear_sparse_input)
    linear_logit += Linear(False, linear_l2_reg)(linear_dense_input)

    # deep part
    dnn_emb_list, dnn_dense_list = get_input_from_feat_columns(input_feats, feat_columns, dnn_l2_reg, prefix='Dnn_Cin')
    dnn_emb_input = tf.concat(dnn_emb_list, axis=1)
    dnn_dense_input = tf.concat(dnn_dense_list, axis=1)

    dnn_input = tf.concat([keras.layers.Flatten()(dnn_emb_input), dnn_dense_input], axis=1)

    dnn_output = DNN(
        hidden_units=dnn_hidden_units,
        activation=dnn_activation,
        l2_reg=dnn_l2_reg,
        dropout_rate=dnn_dropout_rate,
        use_bn=dnn_use_bn
    )(dnn_input)
    dnn_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    # CIN part
    cin_output = CIN(cin_layers, cin_split, cin_activation, dnn_l2_reg)(dnn_emb_input)
    cin_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(cin_output)

    logit = linear_logit + dnn_logit + cin_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model