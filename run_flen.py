import sys, os

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(__file__))
os.chdir(sys.path[0])


import pandas as pd
from tensorflow.python import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from main_code.feature import SparseFeat
from main_code.model import FLEN
from main_code.utils import set_seeds
from datetime import datetime
from copy import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    model_name = 'FLEN'
    log_dir = os.path.join('log', 'tensorboard', model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    task = 'binary'
    emb_size = 16
    test_ratio = 0.2
    linear_l2_reg = 0.0
    dnn_hidden_units = (32,16,8)
    dnn_activation = 'relu'
    dnn_l2_reg = 0.0
    dnn_use_bn = True
    dnn_dropout_rate = 0.0
    lr = 0.001
    train_batch_size = 256
    test_batch_size = 512
    epochs=100
    validation_split=0.1

    set_seeds(seed=2023)

    data = pd.read_csv('datasets/avazu_sample.txt')
    data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
    data['hour'] = data['hour'].apply(lambda x: str(x)[6:])

    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'device_type', 'device_conn_type', 'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]

    data[sparse_features] = data[sparse_features].fillna('-1')
    target = ['click']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    field_info = dict(C14='user', C15='user', C16='user', C17='user',
                      C18='user', C19='user', C20='user', C21='user', C1='user',
                      banner_pos='context', site_id='context',
                      site_domain='context', site_category='context',
                      app_id='item', app_domain='item', app_category='item',
                      device_model='user', device_type='user',
                      device_conn_type='context', hour='context',
                      device_id='user'
                      )

    feat_columns = [SparseFeat(name=feat_name, emb_size=emb_size, vocab_size=data[feat_name].max()+1, group_name=group_name) for feat_name, group_name in field_info.items()]

    linear_feat_columns = copy(feat_columns)
    dnn_feat_columns = copy(feat_columns)

    train, test = train_test_split(data, test_size=0.2)

    train_input = {feat: train[feat] for feat in field_info.keys()}
    test_input = {feat: test[feat] for feat in field_info.keys()}

    model = FLEN(linear_feat_columns, dnn_feat_columns, task, linear_l2_reg, dnn_hidden_units, dnn_activation, dnn_l2_reg, \
        dnn_use_bn, dnn_dropout_rate)

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
        x=train_input, 
        y=train[target].values, 
        batch_size=train_batch_size,
        epochs=epochs,
        verbose=2,
        validation_split=validation_split,
        callbacks=cbk_list
    )

    preds = model.predict(x=test_input, batch_size=test_batch_size)

    print('test LogLoss', round(log_loss(test[target].values, preds), 4))
    print('test AUC', round(roc_auc_score(test[target].values, preds), 4))