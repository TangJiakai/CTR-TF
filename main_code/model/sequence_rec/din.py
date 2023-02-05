import tensorflow as tf
from tensorflow.python import keras
from main_code.inputs import build_input_feats
from main_code.feature import SparseFeat, VarLenSparseFeat
from main_code.inputs import create_emb_dict
from main_code.layer import DINAttentionLayer, DNN, PredictLayer



def DIN(feat_columns, attn_hidden_units, dnn_hidden_units, dnn_activation, dnn_dropout_rate, dnn_l2_reg,
        dnn_use_bn, task, max_len, prefix='DIN'):
    
    input_feats = build_input_feats(feat_columns)

    sparse_feat_columns = list(filter(lambda x:isinstance(x, SparseFeat), feat_columns))
    hist_feat_columns = list(filter(lambda x:isinstance(x, VarLenSparseFeat), feat_columns))
    
    sparse_feat_dict = create_emb_dict(sparse_feat_columns, dnn_l2_reg, mask_zero=True, prefix='sparse_')
    sparse_feat_input = {}

    for feat in sparse_feat_columns:
        feat_id = input_feats[feat.name]
        feat_emb = sparse_feat_dict[feat.emb_name](feat_id)
        sparse_feat_input[feat.name] = feat_emb

    for feat in hist_feat_columns:
        hist_feat_ids = input_feats[feat.name]
        sparse_feat_input[feat.name] = sparse_feat_dict[feat.emb_name](hist_feat_ids)

    sparse_feat_input['target_item'] = tf.concat([sparse_feat_input['target_item'], sparse_feat_input['target_cate']], axis=2)

    mask = tf.sequence_mask(tf.convert_to_tensor(input_feats['hist_len']), maxlen=max_len)
    sparse_feat_input['hist_item'] = tf.concat([sparse_feat_input['hist_item'],sparse_feat_input['hist_cate']], axis=2)

    hist_pooing_input = DINAttentionLayer(attn_hidden_units)([sparse_feat_input['target_item'], sparse_feat_input['hist_item']], mask=mask)

    dnn_input = tf.concat([sparse_feat_input['user_id'], hist_pooing_input, sparse_feat_input['target_item']], axis=2)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(dnn_input)
    logit = keras.layers.Dense(1,  use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    output = PredictLayer(task)(logit)
    model = keras.Model(inputs=input_feats, outputs=output)

    return model

    

    