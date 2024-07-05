# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:46:32 2024

@author: BM109X32G-10GPU-02
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:25:57 2021
@author: BM109X32G-10GPU-02
"""

import pandas as pd
import matplotlib
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error,r2_score, mean_absolute_error,mean_squared_error
import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import preprocess as pp
import pickle

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)
        self.W_mlp = nn.Linear(dim*2, dim)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        Smiles,fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        if len(fingerprints) !=  len(adjacencies):
            zero=torch.LongTensor(np.zeros(((len(adjacencies)-len(fingerprints))))).to(device)
            fingerprints =torch.cat( [fingerprints,zero])
        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)

        for l in range(layer_hidden):

            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        return Smiles,molecular_vectors

    def mlp(self, vectors):
        """ regressor based on multilayer perceptron."""
        vectors = torch.relu(self.W_mlp(vectors)) 
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs
    def forward_regressor(self, data_batch1,data_batch2, train):

        inputs1 = data_batch1[:-1]
        inputs2 = data_batch2[:-1]
        correct_values = torch.cat(data_batch1[-1])

        if train:
            Smiles1,molecular_vectors1 = self.gnn(inputs1)
            Smiles2,molecular_vectors2 = self.gnn(inputs2)
            molecular_vectors1 = molecular_vectors1.to('cpu').data.numpy()
            molecular_vectors2 = molecular_vectors2.to('cpu').data.numpy()
            molecular_vectors = np.concatenate((molecular_vectors1,molecular_vectors2),axis=1)

            molecular_vectors= torch.tensor(molecular_vectors).to(device)

            predicted_values = self.mlp(molecular_vectors)
            #print(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            with torch.no_grad():
                 Smiles1,molecular_vectors1 = self.gnn(inputs1)
                 Smiles2,molecular_vectors2 = self.gnn(inputs2)
                 molecular_vectors1 = molecular_vectors1.to('cpu').data.numpy()
                 molecular_vectors2 = molecular_vectors2.to('cpu').data.numpy()                 
                 molecular_vectors = np.concatenate((molecular_vectors1,molecular_vectors2),axis=1) 
                 molecular_vectors= torch.tensor(molecular_vectors).to(device)
                 predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles1,predicted_values, correct_values


        return Smiles1,predicted_values

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset1,dataset2):

        N1 = len(dataset1)
        N2 = len(dataset2)
        loss_total = 0
        for i in range(0, N1, batch_train):
            data_batch1 = list(zip(*dataset1[i:i+batch_train]))
            data_batch2 = list(zip(*dataset2[i:i+batch_train]))
            loss = self.model.forward_regressor(data_batch1,data_batch2, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model
    def test_regressor(self, dataset1,dataset2):
        N = len(dataset1)
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch1 = list(zip(*dataset1[i:i+batch_test]))
            data_batch2 = list(zip(*dataset2[i:i+batch_test]))
            (Smiles,  predicted_values,correct_values) = self.model.forward_regressor(
                                               data_batch1,data_batch2, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_values)
            Ys.append(predicted_values)

            SAE += sum(np.abs(predicted_values-correct_values))
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        #MSE = SE_sum / N
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        MAEs = SAE / N  # mean absolute error.
        return MAEs,predictions

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]
def dump_dictionary(dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = [np.arange(3)]
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(3)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
if __name__ == "__main__":
    radius=1
    dim=52
    layer_hidden=2
    layer_output=5
    batch_train=50
    batch_test=50
    lr=1e-3
    lr_decay=0.95
    decay_interval=20
    iteration=1000
    N=3000
    path=r'J:/libray/DeepDAP/discussion/'
    dataname=''
    device = torch.device('cuda')
    import datetime
    time1=str(datetime.datetime.now())[0:13]
    dataset_traina,dataset_traind = pp.create_dataset('data.txt',path,dataname)

    dataset_testa,dataset_testd = pp.create_dataset('test.txt',path,dataname)
   # dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)  

    lr, lr_decay = map(float, [lr, lr_decay])
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     print('The code uses a GPU!')
    # else:
    #     device = torch.device('cpu')
    #     print('The code uses a CPU...')

    torch.manual_seed(111)
    model = MolecularGraphNeuralNetwork(
            N, dim, layer_hidden, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_MAEs = path+ 'PCE_MAEs'+'.txt'
    file_test_result  = path+  'PCE_test_prediction'+ '.txt'
    file_dev_result  = path+ 'PCE_val_prediction'+ '.txt'
    file_train_result  = path+ 'PCE_train_prediction'+ '.txt'
    file_model = path+  'PCE_model'+'.h5'
    file1= path + 'PCE-MAE.png'
    file2= path + 'PCE-train.png'
    file3= path + 'PCE-test.png'
    file4= path + 'PCE-val.png'

    result = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMAE_dev'  
    #tMAE_test
    with open(file_MAEs, 'w') as f:
        f.write(result + '\n') 
    print('Start training.')
    print('The result is saved in the output directory every epoch!')
    start = timeit.default_timer()
    for epoch in range(iteration):
        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        model.train()
        loss_train = trainer.train(dataset_traina,dataset_traind)
        MAE_tf_best=9999999
        model.eval()
        MAE_train,predictions_train = tester.test_regressor(dataset_traina,dataset_traind)
        MAE_dev = tester.test_regressor(dataset_testa,dataset_testd)[0]

        time = timeit.default_timer() - start
        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                   hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)
        result = '\t'.join(map(str, [epoch, time, loss_train, MAE_train, MAE_dev])) 
        tester.save_MAEs(result, file_MAEs)
        if MAE_dev <= MAE_tf_best:
            MAE_tf_best = MAE_dev
           # tester.save_model(model, file_model)
        print(result)

    loss = pd.read_table(file_MAEs)
    plt.plot(loss['MAE_train'], color='b',label='MSE of train set')
    plt.plot(loss['MAE_dev'], color='y',label='MSE of validation set')
    #plt.plot(loss['MAE_test'], color='green',label='MSE of test set')
    plt.ylabel('PCELoss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(file1,dpi=300)
    plt.show()


    predictions_test = tester.test_regressor(dataset_testa,dataset_testd)[1]
    tester.save_predictions(predictions_test, file_test_result)


    # r2 = r2_score(res ['Correct'], res ['Predict'])
    # mae = mean_absolute_error(res ['Correct'], res ['Predict'])
    # medae = median_absolute_error(res ['Correct'], res ['Predict'])
    # rmae = np.mean(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct']) * 100
    # median_re = np.median(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct'])
    # mean_re=np.mean(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct'])
    # plt.plot(res ['Correct'], res ['Predict'], '.', color = 'yellow')
    # plt.plot([4,12], [4,12], color ='red')
    # plt.ylabel('Predicted PCE')
    # plt.xlabel('Experimental PCE')        
    # plt.text(4,12, 'R2='+str(round(r2,4)), fontsize=12)
    # plt.text(6,11,'MAE='+str(round(mae,4)),fontsize=12)
    # plt.text(8, 10, 'MedAE='+str(round(medae,4)), fontsize=12)
    # plt.text(4, 11, 'MRE='+str(round(mean_re,4)), fontsize=12)
    # plt.text(6, 12, 'MedRE='+str(round(median_re,4)), fontsize=12)
    # plt.savefig( path+expain+time1+'PCE-dev.tif',dpi=300)
    # plt.figure()
    # plt.show()

    res  = pd.read_table(file_test_result)
    r2 = r2_score(res ['Correct'], res ['Predict'])
    mae = mean_absolute_error(res ['Correct'], res ['Predict'])
    medae = median_absolute_error(res ['Correct'], res ['Predict'])
    rmae = np.mean(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct']) * 100
    median_re = np.median(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct'])
    mean_re=np.mean(np.abs(res ['Correct'] - res ['Predict']) / res ['Correct'])
    mse = mean_squared_error(res ['Correct'], res ['Predict'])
    plt.plot(res ['Correct'], res ['Predict'], '.', color = 'green')
    plt.plot([4,12], [4,12], color ='red')
    plt.ylabel('Predicted PCE')
    plt.xlabel('Experimental PCE')        
    plt.text(4,12, 'R2='+str(round(r2,4)), fontsize=12)
    plt.text(6,11,'MAE='+str(round(mae,4)),fontsize=12)
    plt.text(8, 10, 'MedAE='+str(round(medae,4)), fontsize=12)
    plt.text(4, 11, 'MRE='+str(round(mean_re,4)), fontsize=12)
    plt.text(6, 12, 'MedRE='+str(round(median_re,4)), fontsize=12)
    plt.savefig( path+expain+time1+'PCE-Test.tif',dpi=300)
    plt.figure()
    plt.show()
    from scipy.stats import pearsonr
    print(pearsonr(res ['Correct'], res ['Predict']))