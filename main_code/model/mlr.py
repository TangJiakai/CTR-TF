import tensorflow as tf
from tensorflow.python import keras
from main_code.feature import SparseFeat
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.layer import Linear, PredictLayer
from copy import copy

def get_linear_feat_columns(feat_columns):
    for i in range(len(feat_columns)):
        feat = feat_columns[i]
        if isinstance(feat, SparseFeat):
            feat_columns[i] = feat_columns[i]._replace(emb_size=1, emb_initializer=keras.initializers.Zeros())

    return feat_columns


def get_linear_score(input_feats, feat_columns, l2_reg, emb_reg, prefix=''):
    emb_list, dense_list = get_input_from_feat_columns(input_feats, feat_columns, emb_reg, prefix=prefix)
    emb_feat = tf.concat(emb_list, axis=1)
    dense_feat = tf.concat(dense_list, axis=1)

    logit = tf.reduce_sum(emb_feat)
    logit += Linear(False, l2_reg)(dense_feat)

    return logit


def MLR(region_feat_columns, base_feat_columns, task, region_num, bias_feat_columns=None, l2_reg=0., emb_l2_reg=0.0, prefix='MLR'):
    if base_feat_columns is None:
        base_feat_columns = copy(region_feat_columns)
    
    if bias_feat_columns is None:
        bias_feat_columns = []

    input_feats = build_input_feats(region_feat_columns + base_feat_columns + bias_feat_columns)

    region_feat_columns = get_linear_feat_columns(region_feat_columns)
    base_feat_columns = get_linear_feat_columns(base_feat_columns)
    bias_feat_columns = get_linear_feat_columns(bias_feat_columns)
    
    region_score_list = [get_linear_score(input_feats, region_feat_columns, l2_reg, emb_l2_reg, prefix=prefix+'_region'+str(i)) for i in range(region_num)]
    region_scores = tf.concat(region_score_list, axis=-1)
    region_scores = keras.layers.Softmax()(region_scores)

    learner_score_list = [get_linear_score(input_feats, base_feat_columns, l2_reg, emb_l2_reg, prefix=prefix+'_learner'+str(i)) for i in range(region_num)]
    learner_scores = tf.concat(learner_score_list, axis=-1)

    logit = keras.layers.dot([region_scores, learner_scores], axes=1)

    if len(bias_feat_columns) > 0:
        bias_output = get_linear_score(input_feats, bias_feat_columns, l2_reg, emb_l2_reg, prefix=prefix+'_bias')
        logit *= bias_output

    output = PredictLayer(task)(logit)
    model = keras.Model(inputs=input_feats, outputs=output)

    return model
    

    