from tensorflow.python import keras
from main_code.feature import SparseFeat, DenseFeat, VarLenSparseFeat
from collections import defaultdict, OrderedDict


def create_emb_dict(feat_columns, l2_reg, mask_zero, prefix='sparse_'):
    emb_dict = {}
    for feat in feat_columns:
        emb = keras.layers.Embedding(
            input_dim=feat.vocab_size,
            output_dim=feat.emb_size,
            embeddings_initializer=feat.emb_initializer,
            embeddings_regularizer=keras.regularizers.l2(l2_reg),
            mask_zero=mask_zero,
            name=prefix + '_emb_' + feat.emb_name
        )
        emb.trainable = feat.trainable
        emb_dict[feat.emb_name] = emb

    return emb_dict


def emb_lookup(emb_dict, input_feat, sparse_feat_columns):
    emb_result = defaultdict(list)

    for feat in sparse_feat_columns:
        feat_name = feat.name
        feat_id = input_feat[feat_name]
        emb_matrix = emb_dict[feat_name]
        emb_result[feat_name] = emb_matrix(feat_id)
    
    return emb_result

def group_emb_lookup(emb_dict, input_feat, sparse_feat_columns):
    group_emb_result = {}

    for feat in sparse_feat_columns:
        feat_name = feat.name
        feat_id = input_feat[feat_name]
        emb_matrix = emb_dict[feat_name]
        if feat.group_name not in group_emb_result:
            group_emb_result[feat.group_name] = {}
        group_emb_result[feat.group_name][feat.name] = emb_matrix(feat_id)
    
    return group_emb_result


def get_dense_input(input_feat, feat_columns):
    dense_result = []

    for feat in feat_columns:
        dense_result.append(input_feat[feat.name])
    
    return dense_result


def build_input_feats(feat_columns):
    input_feats = OrderedDict()

    for feat in feat_columns:
        if isinstance(feat, SparseFeat):
            input_feats[feat.name] = keras.layers.Input(shape=(1,), name=feat.name, dtype=feat.dtype)
        elif isinstance(feat, DenseFeat):
            input_feats[feat.name] = keras.layers.Input(shape=(feat.dimension,), name=feat.name, dtype=feat.dtype)
        elif isinstance(feat, VarLenSparseFeat):
            input_feats[feat.name] = keras.layers.Input(shape=(feat.max_len), name=feat.name, dtype=feat.dtype)

    return input_feats


def get_input_from_feat_columns(input_feats, feat_columns, l2_reg, mask_zero=False, prefix=''):
    sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feat_columns))
    dense_feat_columns = list(filter(lambda x: isinstance(x, DenseFeat), feat_columns))

    emb_dict = create_emb_dict(sparse_feat_columns, l2_reg, mask_zero, prefix)
    sparse_emb_dict = emb_lookup(emb_dict, input_feats, sparse_feat_columns)
    sparse_emb_input = list(sparse_emb_dict.values())

    dense_input = get_dense_input(input_feats, dense_feat_columns)

    return sparse_emb_input, dense_input
    

def get_input_from_group_feat_columns(input_feats, feat_columns, l2_reg, mask_zero=False, prefix=''):
    sparse_feat_columns = list(filter(lambda x: isinstance(x, SparseFeat), feat_columns))
    dense_feat_columns = list(filter(lambda x: isinstance(x, DenseFeat), feat_columns))

    emb_dict = create_emb_dict(sparse_feat_columns, l2_reg, mask_zero, prefix=prefix)
    sparse_group_emb_dict = group_emb_lookup(emb_dict, input_feats, sparse_feat_columns)

    dense_input = get_dense_input(input_feats, dense_feat_columns)

    return sparse_group_emb_dict, dense_input