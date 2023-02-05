import tensorflow as tf
from tensorflow.python import keras
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from main_code.layer import PredictLayer, DNN, InnerProduct, OuterProduct
from copy import copy


def PNN(feat_columns, task, D, pnn_type, emb_l2_reg=0.0, dnn_hidden_units=(32,16), dnn_activation='relu', dnn_use_bn=False, dnn_dropout_rate=0.5, 
     dnn_l2_reg=0.5, prefix='FM'):

    input_feats = build_input_feats(feat_columns)

    emb_list, dense_list = get_input_from_feat_columns(input_feats, feat_columns, emb_l2_reg, prefix='PNN')
    emb_input = tf.concat(emb_list, axis=1)    
    dense_input = tf.concat(dense_list, axis=1)

    Lz_input = keras.layers.Flatten()(tf.concat(emb_list, axis=1))
    Lz = keras.layers.Dense(D, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(Lz_input)
    L_p = InnerProduct(D, dnn_l2_reg)(emb_input) if pnn_type == 'inner' else OuterProduct(D, dnn_l2_reg)(emb_input)
    L_input = tf.concat([Lz, L_p, dense_input], axis=1)
    L_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(L_input)
    output = keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(L_output)

    logit = PredictLayer(task)(output)

    model = keras.Model(inputs=input_feats, outputs=logit)

    return model