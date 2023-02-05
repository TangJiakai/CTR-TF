from collections import namedtuple
from tensorflow.python import keras


class SparseFeat(namedtuple('SparseFeat', ['name', 'vocab_size', 'emb_size', 'dtype', 'emb_initializer', 'trainable', 'emb_name', 'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocab_size, emb_size, dtype='int32', emb_initializer=None, trainable=True, emb_name=None, group_name='default'):
        emb_initializer = keras.initializers.random_normal(mean=0, stddev=0.0001)

        if emb_name is None:
            emb_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocab_size, emb_size, dtype, emb_initializer, trainable, emb_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['sparse_feat', 'max_len', 'emb_name'])):
    __slots__ = ()

    def __new__(cls, sparse_feat, max_len, emb_name):
        return super(VarLenSparseFeat, cls).__new__(cls, sparse_feat, max_len, emb_name)
    
    @property
    def name(self):
        return self.sparse_feat.name

    @property
    def emb_size(self):
        return self.sparse_feat.emb_size
    
    @property
    def vocab_size(self):
        return self.sparse_feat.vocab_size
    
    @property
    def dtype(self):
        return self.sparse_feat.dtype
    
    @property
    def emb_initializer(self):
        return self.sparse_feat.emb_initializer
    
    @property
    def trainable(self):
        return self.sparse_feat.trainable

    @property
    def group_name(self):
        return self.sparse_feat.group_name

    def __hash__(self):
        return self.name.__hash__()

    

    




    