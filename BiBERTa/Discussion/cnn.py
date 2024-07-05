# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:48:02 2024

@author: BM109X32G-10GPU-02
"""
 
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from scipy import sparse
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score,median_absolute_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#from smiles_to_onehot.encoding import get_dict, one_hot_coding
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #Disable Tensorflow debugging information
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Set GPU device -1


def split_smiles(smiles, kekuleSmiles=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
    except:
        pass
    splitted_smiles = []
    for j, k in enumerate(smiles):
        if len(smiles) == 1:
            return [smiles]
        if j == 0:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            else:
                splitted_smiles.append(k)
        elif j != 0 and j < len(smiles) - 1:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            elif k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)

        elif j == len(smiles) - 1:
            if k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
    return splitted_smiles

def get_maxlen(all_smiles, kekuleSmiles=True):
    maxlen = 0
    for smi in tqdm(all_smiles):
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        maxlen = max(maxlen, len(spt))
    return maxlen
def get_dict(all_smiles, save_path, kekuleSmiles=True):
    words = [' ']
    for smi in tqdm(all_smiles):
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        for w in spt:
            if w in words:
                continue
            else:
                words.append(w)
    with open(save_path, 'w') as js:
        json.dump(words, js)
    return words

def one_hot_coding(smi, words, kekuleSmiles=True, max_len=1000):
    coord_j = []
    coord_k = []
    spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
    if spt is None:
        return None
    for j,w in enumerate(spt):
        if j >= max_len:
            break
        try:
            k = words.index(w)
        except:
            continue
        coord_j.append(j)
        coord_k.append(k)
    data = np.repeat(1, len(coord_j))
    output = sparse.csr_matrix((data, (coord_j, coord_k)), shape=(max_len,600))
    return output


if __name__ == "__main__":
    data_train= pd.read_csv(r"J:\libray\DeepDAP\DeepDAP\dataset\OSC\train.csv")
    data_test=pd.read_csv(r"J:\libray\DeepDAP\DeepDAP\dataset\OSC\test.csv")
    dono = list(data_train['donor'])
    acce= list(data_train['acceptor'])
    rts = list(data_train['Label'])

    smiles, targets = [], []
    for i, inc in enumerate(tqdm(dono)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)
            targets.append(rts[i])

    words = get_dict(smiles, save_path='dict.json')

    features = []
    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=1000)
        if xi is not None:
            features.append(xi.todense())
    acceptor,smiles =  [], []
    for i, inc in enumerate(tqdm(acce)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)

    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=1000)
        if xi is not None:
            acceptor.append(xi.todense())    

    features = np.asarray(features)
    acceptor = np.asarray(acceptor)
    targets = np.asarray(targets)
    don=features
    acc=acceptor
    X_train=np.dstack((don,acc))
    Y_train=targets


   # physical_devices = tf.config.experimental.list_physical_devices('CPU') 
   # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  #  tf.config.experimental.set_memory_growth(physical_devices[0], True)


    dono = list(data_test['donor'])
    acce= list(data_test['acceptor'])
    rts = list(data_test['Label'])

    smiles, targets = [], []
    for i, inc in enumerate(tqdm(dono)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)
            targets.append(rts[i])
    features = []
    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=1000)
        if xi is not None:
            features.append(xi.todense())            

    acceptor,smiles =  [], []
    for i, inc in enumerate(tqdm(acce)):
        mol = Chem.MolFromSmiles(inc)
        if mol is None:
            continue
        else:
            smi = Chem.MolToSmiles(mol)
            smiles.append(smi)

    for i, smi in enumerate(tqdm(smiles)):
        xi = one_hot_coding(smi, words, max_len=1000)
        if xi is not None:
            acceptor.append(xi.todense())    


    features = np.asarray(features)
    targets = np.asarray(targets)
    acceptor = np.asarray(acceptor)
    don=features
    acc=acceptor
    X_test=np.dstack((don,acc))
    Y_test=targets

    layer_in = Input(shape=(X_train.shape[1:3]), name="smile")
    layer_conv = layer_in
    for i in range(3):
            layer_conv = Conv1D(128, kernel_size=4, activation='relu', kernel_initializer='normal')(layer_conv)
            layer_conv = MaxPooling1D(pool_size=3)(layer_conv)
    layer_dense = Flatten()(layer_conv)

    for i in range(3):
        layer_dense = Dense(32, activation="relu", kernel_initializer='normal')(layer_dense)
    layer_output = Dense(1, activation="relu", name="output")(layer_dense)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')
    mcp_save = ModelCheckpoint('cnn.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, verbose=1, min_delta=1e-4, mode='min')

    model = Model(layer_in, outputs = layer_output) 
    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[metrics.mae])
    from tensorflow.keras import backend as K #转换为张量
    X_train = K.cast_to_floatx(X_train)
    Y_train = K.cast_to_floatx(Y_train)
    history = model.fit(X_train, Y_train, epochs=200, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.01)
    X_test = K.cast_to_floatx( X_test)
    Y_test= K.cast_to_floatx(Y_test)
    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['loss', 'mae', 'val_loss', 'val_mae'], loc='upper left')
    plt.show()
    # plt.savefig("Result/retention_" + save_name + '_loss.png')

    # predict
    model = load_model('cnn.h5')
    Y_predict = model.predict(X_test)
    Y_predict=Y_predict.reshape(len(Y_predict))
     #Y_predict = model.predict(X_test)#训练数据
    x = list(Y_test)
    y = list(Y_predict)

    from scipy.stats import pearsonr
    print(pearsonr(x,y))
    from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
    print(pearsonr(x,y))

    r2 = r2_score(x,y)
    mae = mean_absolute_error(x,y)
    mse = mean_squared_error(x,y)
    r2 = r2_score(x,y)
    mae = mean_absolute_error(x,y)
    medae = median_absolute_error(x,y)