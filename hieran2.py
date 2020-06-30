import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
#from torchvision import datasets, transforms
from torch.autograd import Variable
#import matplotlib.pyplot as plt
from model_hieran import VRNN
import numpy
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import xlsxwriter
from sklearn.model_selection import train_test_split



if not os.path.exists('savesg10'):
    os.makedirs('savesg10')

def train(epoch,data,num,seed):
    look_back = 60
    data1 = Variable(torch.tensor(data).squeeze(0))
    data1 = data1.float()
    print('data shape', data1.shape)

    torch.manual_seed(seed)
    #plt.ion()

    
   
    x_dim = num_features
    kld_loss, nll_loss, pre =  model_train(data1)
    print('nll-loss',nll_loss)
    print('kld-loss',kld_loss)
    print('total', nll_loss+kld_loss)
    tb = nll_loss + kld_loss
    kl.append( np.array(kld_loss.data.detach() ))
    nl.append( np.array(nll_loss.data.detach() ))
    tot.append( np.array(tb.data.detach() ))
    """
    plt.figure(figsize=(15,10));ind = np.linspace(1,len(tot),len(tot))
    #plt.plot(ind, nl, label = 'nll',  ms=12);
    #plt.plot(ind, kl, label = 'kl',   ms=12);
    plt.plot(ind, tot, label = 'tot', ms=12);
    
    plt.legend(loc="upper right")
    plt.xlabel('epochs',fontsize=12, fontweight="bold")
    plt.ylabel('loss', fontsize=12,  fontweight="bold")
    plt.savefig('loss-hiera-tot.eps',format='eps')
    plt.clf()
    plt.figure(figsize=(15,10));ind = np.linspace(1,len(tot),len(tot))
    plt.plot(ind, kl, label = 'kl',   ms=12);

    
    plt.legend(loc="upper right")
    plt.xlabel('epochs',fontsize=12, fontweight="bold")
    plt.ylabel('loss', fontsize=12,  fontweight="bold")
    plt.savefig('loss-hiera-kl.eps',format='eps')
    plt.clf()
    
    plt.figure(figsize=(15,10));ind = np.linspace(1,len(tot),len(tot))
    plt.plot(ind, nl, label = 'nll',  ms=12);

    
    plt.legend(loc="upper right")
    plt.xlabel('epochs',fontsize=12, fontweight="bold")
    plt.ylabel('loss', fontsize=12,  fontweight="bold")
    plt.savefig('loss-hiera-nll.eps',format='eps')
    plt.clf()
    """
    
    
    optimizer.zero_grad()
    err = nll_loss + kld_loss
    #grad norm clipping, only in pytorch version >= 1.10
    err.backward()
    nn.utils.clip_grad_norm_(model_train.parameters(), clip)
    optimizer.step()
    data = np.array(pd.read_csv('lm30.csv',header=None))
    #re = pre - data[look_back:len(data)];N = re.size
    #re = np.sqrt(np.power(re,2));
    #re = np.sum(re)/N
    
    #print(re)
    #pred.append(re)
   
    return model_train



def test(epoch,data):
    data = np.array(pd.read_csv('lm30.csv',header=None))
    #h_dim = data.shape[1]
    #h2_dim = data.shape[1]
    h_dim = 60
    h2_dim = 60
    z_dim = 2
    z2_dim = 10
    n_layers =  1
    n_epochs = 100
    x_dim = data.shape[1]
    data = normalize(data,data.shape[1])
    #_,data = train_test_split(data, test_size=0.2, random_state=5)
    le = math.trunc(0.8*data.shape[0])
    data = data[le:data.shape[0],:]
    
    data = datapro(data,data.shape[1])
    
    path = 'savesg10/hieravrnn_state_dict_'+str(epoch)+'d'+str(n_epochs)+'.pth'
    model_trai = VRNN(x_dim, h_dim, h2_dim, z_dim, z2_dim, n_layers)
    model_trai.load_state_dict(torch.load(path))
    model_trai.eval()
    data1 = Variable(torch.tensor(data).squeeze(0))
    data1 = data1.float()
   
   
    _,_,re = model_trai(data1)
    return re



parser = argparse.ArgumentParser()
data = np.array(pd.read_csv('lm30.csv',header=None))
#data = np.array(data[:,0:2])
#hyperparameters
parser.add_argument('--h_dim', type=int, default=10)
args = parser.parse_args()
h_dim = 60
h2_dim = 60
z_dim = 2
z2_dim = 10
n_layers =  1
n_epochs = 100
clip = 10
learning_rate = 1e-2
x_dim = data.shape[1]
sd = [100,101,102,103,104,105,106,107,108,109]



kl = [];nl = [];tot = [];

model_train = VRNN(x_dim, h_dim, h2_dim, z_dim, z2_dim, n_layers)
optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)


#manual seed
def train_t(dataa,num_features,seed):
    for epoch in range(1, n_epochs + 1):
        model_train = train(epoch,dataa,num_features,seed)
        if epoch % n_epochs == 0:
            """
            plt.plot(pred);plt.show()
            plt.savefig("image-vrnn-new")
            """
        #saving model
        if epoch % n_epochs == 0 or epoch % n_epochs == epoch:
            fn = 'savesg10/hieravrnn_state_dict_'+str(epoch)+'d'+str(n_epochs)+'.pth'
            torch.save(model_train.state_dict(), fn)
            print('Saved model to' + fn)
            #data1 = datapro(data,data.shape[1])
            pre = test(epoch, dataa)
            #pre =  unnormalize(pre,num_features)
           
           
          
            da = np.array(pd.read_csv('lm30.csv',header=None));da = normalize(da,da.shape[1])
            #_,da = train_test_split(da, test_size=0.2, random_state=5)
            le = math.trunc(0.8*da.shape[0])
            da = da[le:da.shape[0],:]
            
         
 
            re = pre - da[look_back:len(da),:];N = re.size
            re = np.sqrt( np.sum(np.power(re,2) )/N );
            #re = np.sum(re)/N
            print(re)
            pred.append(re)
            #print(pred)
            """
            plt.figure(figsize=(15,10));ind = np.linspace(1,len(tot),len(tot))
            
            plt.plot(ind, pred, label = 'mse', ms=12);
               
            plt.legend(loc="upper right")
            plt.xlabel('epochs',fontsize=12, fontweight="bold")
            plt.ylabel('loss', fontsize=12,  fontweight="bold")
            plt.savefig('loss-hiera2-mse.eps',format='eps')
            """
            
            
    return pred







def datapro(Xs,num_features):
    look_back = 60;
    nb_samples = Xs.shape[0] - look_back
    Xtrain2 = np.zeros((nb_samples,look_back,num_features))

    for i in range(nb_samples):
        y_position = i + look_back
        Xtrain2[i] = Xs[i:y_position]
    Xs = Xtrain2
    return (Xs)
    
"""
def normalize(data,num_features):
    data_ = []
    for col in range(0,num_features):
        df = ( data[:,col] - data[:,col].mean() ) / ( data[:,col].std() )
        if col == 0:
            data_ = df.reshape(len(df),1)
        else:
            data_ = np.hstack((data_ , df.reshape(len(df),1) ))
    return data_
"""


def normalize(data, num_features):
    return ( data - np.min(data) ) / (np.max(data)-np.min(data))
    
    
def unnormalize(data,num_features):
    data_ = []
    for col in range(0,num_features):
        df =  data[:,col]*( data[:,col].std() ) /  ( data[:,col] - data[:,col].mean() )
        if col == 0:
            data_ = df.reshape(len(df),1)
        else:
            data_ = np.hstack(( data_ , df.reshape(len(df), 1)  ))
    return data_


if __name__ == '__main__':
    look_back = 60; pred = [];
    #data = np.random.rand(20000,12)
    for iter in range(0,10):
        look_back = 60; pred = [];kl = [];nl = [];tot = [];
        data = np.array(pd.read_csv('lm30.csv',header=None));
        num_features = data.shape[1]
      
        data = normalize(data,num_features)
        #data,_ = train_test_split(data, test_size=0.2, random_state=5)
        le = math.trunc(0.8*data.shape[0])
        data = data[0:le,:]

      
        data = datapro(data,data.shape[1])
      
        model_train = VRNN(x_dim, h_dim, h2_dim, z_dim, z2_dim, n_layers)
        optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)
        pr = train_t(data,num_features,sd[iter])
        if iter == 0:
            ade = np.array(pr).reshape(len(pr),1)
        else:
            ade = np.hstack((ade,np.array(pr).reshape(len(pr),1)))
              
    workbook = xlsxwriter.Workbook(str(z2_dim)+'first'+'vrnn-propencoding.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(ade.T):
        worksheet.write_column(row, col, data)
    workbook.close()
   
