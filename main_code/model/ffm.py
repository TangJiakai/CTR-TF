import tensorflow as tf
from tensorflow.python import keras
from main_code.inputs import build_input_feats, get_input_from_feat_columns
from main_code.feature import SparseFeat
from main_code.layer import PredictLayer, Linear, FFMLayer
from copy import copy


def create_ffm_emb(sparse_feat_columns, l2_reg, prefix):
    emb_dict = {}
    feat_num = len(sparse_feat_columns)
    for feat in sparse_feat_columns:
        emb_list = []
        for i in range(feat_num):
            emb = keras.layers.Embedding(
                input_dim=feat.vocab_size,
                output_dim=feat.emb_size,
                embeddings_initializer=feat.emb_initializer,
                embeddings_regularizer=keras.regularizers.l2(l2_reg),
                name=prefix + '_emb_' + feat.name + 'for' + sparse_feat_columns[i].name
            )
            emb.trainable = feat.trainable
            emb_list.append(emb)
        emb_dict[feat.name] = emb_list

    return emb_dict


def ffm_emb_lookup(emb_dict, input_feats, sparse_feat_columns):
    emb_result = []
    for feat in sparse_feat_columns:
        emb_list = []
        emb_id = input_feats[feat.name]
        for i in range(len(sparse_feat_columns)):
            emb = emb_dict[feat.name][i](emb_id)
            emb_list.append(emb)
        emb_result.append(tf.concat(emb_list, axis=-2))
    
    return emb_result


def FFM(feat_columns, task, linear_l2_reg=0.0, emb_l2_reg=0.0, prefix='FMM'):
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
    sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feat_columns))

    emb_dict = create_ffm_emb(sparse_feat_columns, emb_l2_reg, prefix)
    sparse_emb_list = ffm_emb_lookup(emb_dict, input_feats, sparse_feat_columns)

    sparse_emb_input = tf.stack(sparse_emb_list, axis=-3)

    ffm_logit = FFMLayer()(sparse_emb_input)

    # prediction part
    logit = linear_logit + ffm_logit
    output = PredictLayer(task)(logit)

    model = keras.Model(inputs=input_feats, outputs=output)

    return model