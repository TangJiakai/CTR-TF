import tensorflow as tf
from tensorflow.python import keras
from main_code.layer import Linear, DNN, PredictLayer
from main_code.inputs import build_input_feats, get_input_from_feat_columns, get_dense_input
from main_code.feature import SparseFeat, DenseFeat
from copy import copy
from itertools import combinations


def ONN(linear_feat_columns, dnn_feat_columns, linear_l2_reg=0., dnn_hidden_units=(64,32),
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

    # onn part
    sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feat_columns))
    dense_feat_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feat_columns))
    dense_feat_input = tf.concat(get_dense_input(input_feats, dense_feat_columns), axis=1)

    sparse_embeddings = {}
    for i in range(len(sparse_feat_columns)):
        feat_i = sparse_feat_columns[i]
        tmp_embeddings = {}
        for j in range(len(sparse_feat_columns)):
            feat_j = sparse_feat_columns[j]
            tmp_embeddings[feat_j.name] = keras.layers.Embedding(input_dim=feat_i.vocab_size, output_dim=feat_i.emb_size)
        sparse_embeddings[feat_i.name] = tmp_embeddings

    inner_products = []
    for i, j in combinations(list(range(len(sparse_feat_columns))), 2):
        feat_i_name = sparse_feat_columns[i].name
        feat_j_name = sparse_feat_columns[j].name
        feat_i_id = input_feats[feat_i_name]
        feat_j_id = input_feats[feat_j_name]
        feat_i_emb = tf.squeeze(sparse_embeddings[feat_i_name][feat_j_name](feat_i_id), axis=1)
        feat_j_emb = tf.squeeze(sparse_embeddings[feat_j_name][feat_i_name](feat_j_id), axis=1)
        inner_products.append(keras.layers.dot([feat_i_emb, feat_j_emb], axes=1))

    inner_input = tf.concat(inner_products, axis=1)
    
    copy_results = []
    for i in range(len(sparse_feat_columns)):
        feat_i_name = sparse_feat_columns[i].name
        feat_i_id = input_feats[feat_i_name]
        feat_i_emb = sparse_embeddings[feat_i_name][feat_i_name](feat_i_id)
        copy_results.append(feat_i_emb)

    copy_input = keras.layers.Flatten()(tf.concat(copy_results, axis=1))

    dnn_input = tf.concat([inner_input, copy_input, dense_feat_input], axis=1)
    dnn_input = keras.layers.BatchNormalization()(dnn_input, training=None)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(dnn_input)
    dnn_logit = keras.layers.Dense(1, use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    logit = linear_logit + dnn_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model