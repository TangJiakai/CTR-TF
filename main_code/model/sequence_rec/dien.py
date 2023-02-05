import tensorflow as tf
from tensorflow.python import keras
from main_code.inputs import build_input_feats
from main_code.feature import SparseFeat, VarLenSparseFeat
from main_code.inputs import create_emb_dict
from main_code.layer import DNN, PredictLayer, DIENAttentionLayer


def get_aux_loss(gru1_outputs, hist_item, aux_hist_item, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    pos_prob = tf.sigmoid(keras.layers.dot([gru1_outputs, hist_item], axes=-1))
    neg_prob = tf.sigmoid(keras.layers.dot([gru1_outputs, aux_hist_item], axes=-1))
    aux_loss = -tf.log(pos_prob) * mask
    aux_loss += -tf.log(1 - neg_prob) * mask
    return tf.reduce_mean(aux_loss)


def interest_evolution(attn_hidden_units, l2_reg, target_item, hist_item, aux_hist_item, mask, hist_len):
    gru1_outputs = keras.layers.GRU(units=hist_item.get_shape().as_list()[-1], return_sequences=True, kernel_regularizer=keras.regularizers.l2(l2_reg))(hist_item, mask=tf.squeeze(mask, axis=1))
    mask = tf.transpose(mask, perm=(0,2,1))
    aux_loss = get_aux_loss(gru1_outputs[:,:-1,:], hist_item[:,1:,:], aux_hist_item[:,:-1,:], mask[:,1:,:])
    hist_pooling_output = DIENAttentionLayer(attn_hidden_units, l2_reg)([target_item, gru1_outputs], mask=mask, hist_len=hist_len, name='AUGRU')

    return hist_pooling_output, aux_loss


def DIEN(feat_columns, alpha, attn_hidden_units, dnn_hidden_units, dnn_activation, dnn_dropout_rate, dnn_l2_reg,
        dnn_use_bn, task, max_len, prefix='DIEN'):
    
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
    if len(sparse_feat_input['hist_item'].shape) == 2:
        sparse_feat_input['hist_item'] = tf.expand_dims(sparse_feat_input['hist_item'], axis=1)
        sparse_feat_input['hist_cate'] = tf.expand_dims(sparse_feat_input['hist_cate'], axis=1)
    sparse_feat_input['hist_item'] = tf.concat([sparse_feat_input['hist_item'],sparse_feat_input['hist_cate']], axis=2)

    sparse_feat_input['aux_hist_item'] = tf.concat([sparse_feat_input['aux_hist_item'], sparse_feat_input['aux_hist_cate']], axis=2)

    hist_pooing_input, aux_loss = interest_evolution(attn_hidden_units, dnn_l2_reg, sparse_feat_input['target_item'], sparse_feat_input['hist_item'], sparse_feat_input['aux_hist_item'],\
        mask=mask, hist_len=tf.convert_to_tensor(input_feats['hist_len']))

    dnn_input = tf.concat([sparse_feat_input['user_id'], tf.expand_dims(hist_pooing_input, axis=1), sparse_feat_input['target_item']], axis=2)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, dnn_l2_reg, dnn_dropout_rate, dnn_use_bn)(dnn_input)
    logit = keras.layers.Dense(1,  use_bias=False, kernel_regularizer=keras.regularizers.l2(dnn_l2_reg))(dnn_output)

    output = PredictLayer(task)(logit)
    model = keras.Model(inputs=input_feats, outputs=output)

    model.add_loss(alpha * aux_loss)
    keras.backend.get_session().run(tf.global_variables_initializer())

    return model

    

    