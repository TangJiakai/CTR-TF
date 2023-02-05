import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, DNN, PredictLayer, EDCNCrossNet, BridgeModule, RegulationModule
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from copy import copy


def EDCN(linear_feat_columns, dnn_feat_columns, cross_num, tau, bridge_type, linear_l2_reg=0., emb_l2_reg=0., dcn_l2_reg=0.,
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
    dnn_emb_list, _ = get_input_from_feat_columns(input_feats, dnn_feat_columns, emb_l2_reg, prefix='cross_and_dnn')
    emb_input = tf.concat(dnn_emb_list, axis=1)
    dcn_emb_input = RegulationModule(tau)(emb_input)
    dcn_emb_0 = dcn_emb_input
    dnn_emb_input = RegulationModule(tau)(emb_input)
    field_num, feat_num = int(emb_input.shape[1]), int(emb_input.shape[2])

    for i in range(cross_num):
        dcn_emb_output = EDCNCrossNet(dcn_l2_reg)([dcn_emb_0, dcn_emb_input])
        dnn_emb_output = DNN([field_num * feat_num], dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(dnn_emb_input)
        bridge_output = BridgeModule(bridge_type)([dcn_emb_output, dnn_emb_output])
        if i < cross_num - 1:
            bridge_output = keras.layers.Reshape([field_num, feat_num])(bridge_output)  
            dcn_emb_input = RegulationModule(tau)(bridge_output)
            dnn_emb_input = RegulationModule(tau)(bridge_output)

    linear_input = tf.concat([dcn_emb_output, dnn_emb_output, bridge_output], axis=1)
    cross_and_dnn_logit = Linear(False, dcn_l2_reg)(linear_input)

    # prediction part
    logit = linear_logit + cross_and_dnn_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model
