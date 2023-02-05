import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, DNN, KMaxPooling, PredictLayer
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def CCPM(linear_feat_columns, dnn_feat_columns, conv_kernel_width, conv_filter, linear_l2_reg=0., emb_l2_reg=0., cnn_l2_reg=0., dnn_hidden_units=(64,32),
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

    # dnn part
    dnn_emb_list, dnn_dense_list = get_input_from_feat_columns(input_feats, dnn_feat_columns, emb_l2_reg, prefix='dnn')
    conv_emb_input = tf.concat(dnn_emb_list, axis=1)
    
    conv_num = len(conv_kernel_width)
    feat_num = conv_emb_input.shape[1].value

    conv_output = tf.expand_dims(conv_emb_input, axis=3)

    for i in range(1, conv_num+1):
        width = conv_kernel_width[i-1]
        filter_dim = conv_filter[i-1]

        conv_output = keras.layers.Conv2D(
            filters=filter_dim,
            kernel_size=(width, 1),
            kernel_regularizer=keras.regularizers.l2(cnn_l2_reg),
            padding='same'
        )(conv_output)

        if i < conv_num:
            k = max(1, int((1 - pow(i / conv_num, conv_num - i)) * feat_num))
        else:
            k = 3

        conv_output = KMaxPooling(k)(conv_output)
    
    dnn_input = keras.layers.Flatten()(conv_output)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(dnn_input)

    dnn_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    logit = linear_logit + dnn_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)
    
    return model