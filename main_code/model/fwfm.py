import tensorflow as tf
from tensorflow.python import keras
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from main_code.layer import PredictLayer, Linear, FwFMLayer
from copy import copy


def FwFM(feat_columns, task, linear_l2_reg=0.0, emb_l2_reg=0.0, prefix='FwFM'):
    input_feats = build_input_feats(feat_columns)

    # linear part
    linear_feat_columns = copy(feat_columns)
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
    fwfm_emb_list, fwfm_dense_list = get_input_from_feat_columns(input_feats, feat_columns, emb_l2_reg, prefix=prefix)
    fwfm_logit = FwFMLayer()(fwfm_emb_list)

    # prediction part
    logit = linear_logit + fwfm_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model