import tensorflow as tf
from tensorflow.python import keras
from main_code.feature import SparseFeat
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.layer import Linear


def LR(linear_feat_columns, l2_reg=0.0, prefix='LR'):
    input_feats = build_input_feats(linear_feat_columns)

    for i in range(len(linear_feat_columns)):
        if isinstance(linear_feat_columns[i], SparseFeat):
            linear_feat_columns[i] = linear_feat_columns[i]._replace(emb_size=1, emb_initializer=keras.initializers.zeros())

    linear_emb_list, linear_dense_emb_list = get_input_from_feat_columns(input_feats, linear_feat_columns, l2_reg, prefix)
    sparse_input = tf.concat(linear_emb_list, axis=-1)
    dense_input = tf.concat(linear_dense_emb_list, axis=-1)

    logit = tf.reduce_sum(sparse_input)
    logit += Linear(use_bias=True, l2_reg=l2_reg)(dense_input)
    output = tf.sigmoid(logit)
    
    model = keras.Model(inputs=input_feats, outputs=output)

    return model