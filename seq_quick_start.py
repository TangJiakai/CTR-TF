import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(__file__))
os.chdir(sys.path[0])

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from main_code.feature import SparseFeat, VarLenSparseFeat, DenseFeat
from main_code.model.sequence_rec import DIN, DIEN
from main_code.utils import set_seeds
from datetime import datetime
import pickle


def load_data(path):
    with open(path, 'rb') as f:
        (train_x, train_y) = pickle.load(f)
        (valid_x, valid_y) = pickle.load(f)
        (test_x, test_y) = pickle.load(f)
        (user_num, item_num, cate_num, max_len) = pickle.load(f)
    return train_x, train_y, valid_x, valid_y, test_x, test_y, user_num, item_num, cate_num, max_len


if __name__ == '__main__':

    model_name = 'DIEN'
    log_dir = os.path.join('log', 'tensorboard', model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    task = 'binary'
    emb_size = 16
    attn_hidden_units = (80,40)
    alpha = 1.0
    dnn_hidden_units = (256,128,64)
    dnn_activation = 'relu'
    dnn_l2_reg = 0.0
    dnn_use_bn = True
    dnn_dropout_rate = 0.0
    lr = 0.001
    train_batch_size = 256
    test_batch_size = 512
    epochs = 100

    set_seeds(seed=2023)

    train_x, train_y, valid_x, valid_y, test_x, test_y, user_num, item_num, cate_num, max_length = load_data('datasets/amazon-dien.pkl')

    feat_columns = [SparseFeat(name='user_id', vocab_size=user_num+1, emb_size=emb_size),
                    SparseFeat(name='target_item', vocab_size=item_num+1, emb_size=emb_size, emb_name='item'),
                    SparseFeat(name='target_cate', vocab_size=cate_num+1, emb_size=emb_size, emb_name='cate'),
                    DenseFeat(name='hist_len'),
                    VarLenSparseFeat(sparse_feat=SparseFeat(name='hist_item', vocab_size=item_num+1, emb_size=emb_size, emb_name='item'), max_len=max_length, emb_name='item'),
                    VarLenSparseFeat(sparse_feat=SparseFeat(name='hist_cate', vocab_size=cate_num+1, emb_size=emb_size, emb_name='cate'), max_len=max_length, emb_name='cate'),
                    VarLenSparseFeat(sparse_feat=SparseFeat(name='aux_hist_item', vocab_size=item_num+1, emb_size=emb_size, emb_name='item'), max_len=max_length, emb_name='item'),
                    VarLenSparseFeat(sparse_feat=SparseFeat(name='aux_hist_cate', vocab_size=cate_num+1, emb_size=emb_size, emb_name='cate'), max_len=max_length, emb_name='cate')]

    model = DIEN(
        feat_columns=feat_columns,
        alpha=alpha,
        attn_hidden_units=attn_hidden_units,
        dnn_hidden_units=dnn_hidden_units,
        dnn_activation=dnn_activation,
        dnn_dropout_rate=dnn_dropout_rate,
        dnn_l2_reg=dnn_l2_reg,
        dnn_use_bn=dnn_use_bn,
        task=task,
        max_len=max_length
    )

    model.compile( 
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_crossentropy],
    )
    
    model.summary()

    cbk_list = []

    tensorboard_cbk = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    tensorboard_cbk.set_model(model)

    cbk_list.append(tensorboard_cbk)

    keras.utils.plot_model(model, f'model_images/{model_name}.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    history = model.fit(
        x=train_x, 
        y=train_y, 
        shuffle=True,
        batch_size=train_batch_size,
        validation_data=(valid_x, valid_y),
        epochs=epochs,
        verbose=1,
        callbacks=cbk_list,
    )

    preds = model.predict(x=test_x, batch_size=test_batch_size)

    print('test LogLoss', round(log_loss(test_y, preds), 4))
    print('test AUC', round(roc_auc_score(test_y, preds), 4))
