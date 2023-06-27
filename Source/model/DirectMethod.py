"""
    Implementation of Direct Method

"""

import numpy as np
# import LogisticRegressionModel
from model.RegressionModel import LinearRegressionModel, MLP
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import xgboost as xgb
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F

class DirectMethod:
    def __init__(self, context, treatment, reward):
        self.context = context
        self.treatment = treatment
        self.reward = reward

    def estimate_reward(self, reward, context, action, target_policy, method=None):
        # Hyperparameters
        batch_size = 128
        lr = 0.001
        epochs = 1000

        Train_x = torch.from_numpy(data).float()
        train_y = torch.from_numpy(reward).float()
        dataset = TensorDataset(Train_x, train_y)
        dataLoader = DataLoader(dataset, batch_size, shuffle=True)
        utility = 0
        if method == 'LR':
            utility = self.estimate_by_LR(data=data,
                                          target_policy=target_policy)
            print("Linear Regression:", utility)
        elif method == 'xgb':
            utility = self.estimate_by_xgboost(Train_x=data,
                                               Train_y=reward,
                                               target_policy=target_policy)
            print("xgBoost:", utility)

        elif method == 'mlp':
            utility = self.estimate_by_mlp(data=data,
                                           lr=lr,
                                           reward=reward,
                                           epochs=epochs,
                                           dataLoader=dataLoader,
                                           target_policy=target_policy)
            print("MLP:", utility)

        return utility

    def estimate_by_LR(self, reward, target_policy):
        """
            Direct methods estimation from linear regression.
        """

        data = np.concatenate((self.context, self.treatment), axis=1)

        LRModel = LinearRegression().fit(data, reward)
        dim = data.shape[1]

        predicted = np.zeros_like(target_policy)
        for x in range(target_policy.shape[0]):
            temp = np.zeros(shape=(target_policy.shape[1], dim))
            for t in range(target_policy.shape[1]):
                concat_x = np.append(self.context[x], t)
                temp[t] = concat_x

            predicted[x] = LRModel.predict(temp).squeeze()

        utility = np.average(predicted,
                            weights=target_policy,
                            axis=1)
        return utility.mean()

    def estimate_by_xgboost(self, reward, target_policy):
        """
            Direct method estimation from xgboost
        """
        data = np.concatenate((self.context, self.treatment), axis=1)
        model = xgb.XGBRegressor(random_state=1,nthread=5)
        model.fit(data, reward)

        # Infer the utility value
        dim = data.shape[1]

        predicted = np.zeros_like(target_policy)
        for x in range(target_policy.shape[0]):
            temp = np.zeros(shape=(target_policy.shape[1], dim))
            for t in range(target_policy.shape[1]):
                concat_x = np.append(self.context[x], t)
                temp[t] = concat_x
            predicted[x] = model.predict(temp).squeeze()

        utility = np.average(predicted,
                            weights=target_policy,
                            axis=1)
        return utility.mean()

    def estimate_by_mlp(self, reward, target_policy):
        """
            Direct method estimation by multi-layer perceptron
        """
        data = np.concatenate((self.context, self.treatment), axis=1)

        # MLP by sklearn
        MLPModel = MLPRegressor(random_state=1, max_iter=500).fit(data, reward)
        utility = 0
        dim = data.shape[1]

        predicted = np.zeros_like(target_policy)
        for x in range(target_policy.shape[0]):
            temp = np.zeros(shape=(target_policy.shape[1], dim))
            for t in range(target_policy.shape[1]):
                concat_x = np.append(self.context[x], t)
                temp[t] = concat_x
            predicted[x] = MLPModel.predict(temp).squeeze()

        utility = np.average(predicted,
                            weights=target_policy,
                            axis=1)
        return utility.mean()
    

    def estimate_by_ae(self, reward, target_policy, model, device):
        """
            Direct method estimation by multi-layer perceptron
        """
        train_x_torch = torch.from_numpy(self.context).float().to(device)
        one_hot =  F.one_hot(torch.Tensor([[i] for i in range(target_policy.shape[1])]).long().squeeze()).float()
    

        # Auto-encoder by sklearn
        utility = 0
        predicted = np.zeros_like(target_policy)
        for x in range(target_policy.shape[0]):
            temp_x = train_x_torch[x].repeat(target_policy.shape[1], 1)
            temp, _ = model(temp_x.to(device), one_hot.to(device))
            predicted[x] = temp.detach().cpu().numpy().squeeze()

        utility = np.average(predicted,
                            weights=target_policy,
                            axis=1)
        return utility.mean()



def treatment_transform(t, t_dim):
    if t_dim == 1:
        if t == 0:
            return np.array([0])
        else:
            return np.array([1])
    temp = list('{0:0b}'.format(t))
    treatment = np.zeros(shape=(t_dim, 1))
    temp = [int(i) for i in temp]
    count = 1
    for i in range(len(temp) - 1, -1, -1):
        treatment[len(treatment) - count] = temp[i]
        count += 1
    return np.flip(treatment.squeeze())



