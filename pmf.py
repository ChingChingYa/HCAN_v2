import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState
import copy
from load_data import load_data, read_dataset, cut_data_len
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import matplotlib.pyplot as plt

# https://github.com/mcleonard/pmf-pytorch/blob/master/Probability%20Matrix%20Factorization.ipynb


class PMF(torch.nn.Module):
    def __init__(self, U, V, lambda_U=1e-2, lambda_V=1e-2, latent_size=5,
                 momentum=0.8, learning_rate=0.001, iterations=1000):
        super().__init__()
        torch.manual_seed(91) 
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        # momentum動量  避免曲折過於嚴重
        self.momentum = momentum
        # k數量
        self.latent_size = latent_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.U = None
        self.V = None
        self.R = None
        self.I = None
        self.uv = None
        self.count_users = None
        self.count_items = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       


    def RMSE(self, predicts, truth):
        truth = torch.IntTensor([int(ele) for ele in truth])
        # print(self.count_users)
        # return np.sqrt(np.mean(np.square(predicts - truth)))
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(predicts.float(), truth.float()))
        return loss

    # 原始矩陣分解法目標式
    def loss(self, aspect):
        # the loss function of the model
        if(aspect.shape[0]==0):
            loss = (torch.sum(self.I*(self.R-torch.mm(self.U, self.V.t()))**2)
                    + self.lambda_U*torch.sum(self.U.pow(2))
                    + self.lambda_V*torch.sum(self.V.pow(2)))
        else:
            aspect.requires_grad = True
            b = ((self.uv-aspect)**2)          
            loss = (torch.sum(self.I*(self.R-torch.mm(self.U, self.V.t()))**2)
                    + self.lambda_U*torch.sum(self.U.pow(2))
                    + self.lambda_V*torch.sum(self.V.pow(2))+torch.sum(b))
           
        return loss


    def predict(self, data):
        tmp = torch.LongTensor([0, 1, 2, 3])
        index_data = torch.IntTensor([[int(ele[0]), int(ele[1])] for ele in data])
        u_index = torch.LongTensor([torch.take(i, torch.LongTensor([0])) for i in index_data])
        v_index = torch.LongTensor([torch.take(i, torch.LongTensor([1])) for i in index_data])
        u_features = [torch.gather(self.U[i.item()], 0, tmp) for i in u_index]
        v_features = [torch.gather(self.V[i.item()], 0, tmp) for i in v_index]
        a = [torch.mul(u, v) for u, v in zip(u_features, v_features)]
        preds_value_array = torch.DoubleTensor([torch.sum(i, 0) for i in a])
        uv = [i.detach().numpy() for i in torch.stack(a)]
        return uv, preds_value_array

    def uvcount(self, data):
        tmp = torch.LongTensor([0, 1, 2, 3])
        index_data = torch.IntTensor([[int(ele[0]), int(ele[1])] for ele in data])
        u_index = torch.LongTensor([torch.take(i, torch.LongTensor([0])) for i in index_data])
        v_index = torch.LongTensor([torch.take(i, torch.LongTensor([1])) for i in index_data])
        u_features = [torch.gather(self.U[i.item()], 0, tmp) for i in u_index]
        v_features = [torch.gather(self.V[i.item()], 0, tmp) for i in v_index]
        a = [torch.mul(u, v) for u, v in zip(u_features, v_features)]
        self.uv = torch.stack(a)



    def forward(self, num_users, num_items, train_data=None, valid_data=None,
                aspect_vec=None, U=None, V=None, uv=None, flag=0,
                lambda_U=0.01, lambda_V=0.01):

        train_data = Variable(torch.LongTensor(train_data))
        valid_data = Variable(torch.LongTensor(valid_data))
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V

        # aspect = torch.empty(0)
        aspect = []
        tmp = np.array([0]*6, dtype=np.float32)
        if flag != 0:
            x = int(len(aspect_vec)/0.8*4*0.2)
            # aspect = torch.zeros(x, 6, dtype=torch.float32)
            # for i in range(x):
            #     aspect.append(tmp)
            for i, a in enumerate(aspect_vec):
                for j in range(len(a)):
                    aspect.append(aspect_vec[i][j])

        if self.R is None:
            self.R = torch.zeros(num_users, num_items, dtype=torch.float64)
            for i, ele in enumerate(train_data):
                self.R[int(ele[0]), int(ele[1])] = float(ele[2])
            # 有評分為1 為評分為0
            self.I = copy.deepcopy(self.R)
            self.I[self.I != 0] = 1

        if self.U is None and self.V is None:
            self.count_users = np.size(self.R, 0)
            self.count_items = np.size(self.R, 1)
            # Random
            self.U = torch.rand(self.count_users, self.latent_size,
                                dtype=torch.float64, requires_grad=True)
                                
            self.V = torch.rand(self.count_items, self.latent_size,
                                dtype=torch.float64, requires_grad=True)

            self.uv = torch.rand(self.count_users, self.count_items,
                                 dtype=torch.float64, requires_grad=True)

        else:
            self.U = torch.DoubleTensor(U)
            self.V = torch.DoubleTensor(V)
            self.uv = torch.DoubleTensor(uv)
            self.V.requires_grad = True
            self.U.requires_grad = True
            self.uv.requires_grad = True
        

        optimizer = torch.optim.SGD([self.U, self.V, self.uv],
                                    lr=self.learning_rate,
                                    momentum=self.momentum, nesterov=True)
 
        loss_list = []
        valid_rmse_list = []
        for step, epoch in enumerate(range(self.iterations)):
            optimizer.zero_grad()
            self.uvcount(train_data)
            loss = self.loss(torch.DoubleTensor(aspect))
            loss_list.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

            valid_uv, valid_predicts = self.predict(valid_data)
            valid_rmse = self.RMSE(valid_predicts, valid_data[:, 2])
            valid_rmse_list.append(valid_rmse.detach().numpy())

            if step % 50 == 0:
                print(f'Step {step}\tLoss: {loss:.4f}(train)\tRMSE: {valid_rmse:.4f}(valid)')

        u_list = [i.detach().numpy() for i in self.U]
        v_list = [i.detach().numpy() for i in self.V]
        uv = [i.detach().numpy() for i in self.uv]
        
        loss_list = np.sum(loss_list)
        valid_rmse_list = np.sum(valid_rmse_list)

        return valid_uv, uv, u_list, v_list, loss_list/len(train_data), valid_rmse_list/len(train_data)       
