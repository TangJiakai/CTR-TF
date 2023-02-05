import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(__file__))
os.chdir(sys.path[0])

import pandas as pd
from tensorflow.python import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from main_code.feature import SparseFeat, DenseFeat
from main_code.model import FM, LR, DeepFM, DCN, DCNv2, Fibinet, WDL, CCPM, PNN, MLR, AFM, NFM, xDeepFM, FFM, FwFM, AutoInt, ONN, FGCNN, IFM, FmFM, DIFM, EDCN
from main_code.utils import set_seeds
from datetime import datetime


if __name__ == '__main__':
    model_name = 'EDCN'
    log_dir = os.path.join('log', 'tensorboard', model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    task = 'binary'
    emb_size = 16
    cross_num = 3
    tau = 1.0
    bridge_type = 'attention_pooling'
    linear_l2_reg = 0.0
    emb_l2_reg = 0.0
    lr = 0.001
    dnn_activation = 'relu'
    dnn_l2_reg = 0.0
    dnn_use_bn = False
    dnn_dropout_rate = 0.0
    train_batch_size = 256
    test_ratio = 0.2
    test_batch_size = 512
    epochs=100
    validation_split=0.1

    set_seeds(seed=2023)

    data = pd.read_csv('datasets/criteo_sample.txt')

    sparse_feat = ['C' + str(i) for i in range(1, 27)]
    dense_feat = ['I' + str(i) for i in range(1, 14)]
    
    data[sparse_feat] = data[sparse_feat].fillna('-1')
    data[dense_feat] = data[dense_feat].fillna(0)
    label = ['label']

    for feat in sparse_feat:
        label_encoder = LabelEncoder()
        data[feat] = label_encoder.fit_transform(data[feat])
    
    minmax_scaler = MinMaxScaler()
    data[dense_feat] = minmax_scaler.fit_transform(data[dense_feat])

    feat_columns = [SparseFeat(feat, vocab_size=data[feat].max()+1, emb_size=emb_size) for feat in sparse_feat]
    feat_columns += [DenseFeat(feat, dimension=1) for feat in dense_feat]

    linear_feat_columns = feat_columns
    dnn_feat_columns = feat_columns

    feat_name = sparse_feat + dense_feat

    train_data, test_data = train_test_split(data, test_size=test_ratio)
    train_input = {name: train_data[name] for name in feat_name}
    test_input = {name: test_data[name] for name in feat_name}

    model = EDCN(
        linear_feat_columns=linear_feat_columns,
        dnn_feat_columns=dnn_feat_columns,
        cross_num=cross_num,
        tau=tau,
        bridge_type=bridge_type,
        dnn_activation=dnn_activation,
        dnn_l2_reg=dnn_l2_reg, 
        dnn_use_bn=dnn_use_bn, 
        dnn_dropout_rate=dnn_dropout_rate, 
        linear_l2_reg=linear_l2_reg,
        emb_l2_reg=emb_l2_reg,
        task=task
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
        x=train_input, 
        y=train_data[label].values, 
        batch_size=train_batch_size,
        epochs=epochs,
        verbose=2,
        validation_split=validation_split,
        callbacks=cbk_list
    )

    preds = model.predict(x=test_input, batch_size=test_batch_size)

    print('test LogLoss', round(log_loss(test_data[label].values, preds), 4))
    print('test AUC', round(roc_auc_score(test_data[label].values, preds), 4))
