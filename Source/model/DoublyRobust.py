"""
    Implementation of Doubly Robust method

"""
import numpy as np
from model.RegressionModel import LinearRegressionModel
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import torch.nn.functional as F

class DoublyRobust:
    def __init__(self, context, reward):
        self.context = context
        self.reward = reward

    def estimate_reward(self, reward, action, pscore, target_policy, 
                            group_target_policy, group_action, group_pscore, onehot_t, model,
                            device):
        size = self.context.shape[0]
        action = action.astype('int32')

        # Calculating the inverse_propensity and the rewards
        target_weight = group_target_policy[np.arange(size), np.squeeze(group_action)]
        propensity = group_pscore[np.arange(size), np.squeeze(group_action)]

        inverse_propensity = target_weight / propensity
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = "7"

        # Direct method part
        train_x_torch = torch.from_numpy(self.context).float().to(device)
        one_hot =  F.one_hot(torch.Tensor([[i] for i in range(target_policy.shape[1])]).long().squeeze()).float()

        dm_reward, _ = model(train_x_torch, onehot_t.to(device))
        dm_reward = dm_reward.detach().cpu().numpy()

        predicted = np.zeros_like(target_policy)
        for x in range(target_policy.shape[0]):
            temp_x = train_x_torch[x].repeat(target_policy.shape[1], 1)
            temp, _ = model(temp_x.to(device), one_hot.to(device))
            predicted[x] = temp.detach().cpu().numpy().squeeze()

        res = np.average(predicted, weights=target_policy, axis=1)
        res += ((reward - dm_reward).squeeze() * inverse_propensity)

        print("Doubly Robust: ", res.mean())
        print("Doubly Robust: ", (res / inverse_propensity.mean()).mean() )
        return res


    def estimate_reward_ips(self, reward, action, pscore, target_policy, 
                            onehot_t, model, device):
        size = self.context.shape[0]
        action = action.astype('int32')

        # Calculating the inverse_propensity and the rewards
        target_weight = target_policy[np.arange(size), np.squeeze(action)]
        propensity = pscore[np.arange(size), np.squeeze(action)]

        inverse_propensity = target_weight / propensity

        # Direct method part
        train_x_torch = torch.from_numpy(self.context).float().to(device)
        one_hot =  F.one_hot(torch.Tensor([[i] for i in range(target_policy.shape[1])]).long().squeeze()).float()

        dm_reward, _ = model(train_x_torch, onehot_t.to(device))
        dm_reward = dm_reward.detach().cpu().numpy()

        predicted = np.zeros_like(target_policy)
        for x in range(target_policy.shape[0]):
            temp_x = train_x_torch[x].repeat(target_policy.shape[1], 1)
            temp, _ = model(temp_x.to(device), one_hot.to(device))
            predicted[x] = temp.detach().cpu().numpy().squeeze()

        res = np.average(predicted, weights=target_policy, axis=1)
        res += ((reward - dm_reward).squeeze() * inverse_propensity)

        print("Doubly Robust: ", res.mean())
        print("Doubly Robust: ", (res / inverse_propensity.mean()).mean() )
        return res

