"""
    Regression methods for propensity score calculation
"""
import numpy as np
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB


def cal_propensity_score(context, treatment, device, verbose):
    """
        Return the propensity score: [N * 1]
    """
    # Calculate the propensity using categorical NB
    clf = RandomForestClassifier(random_state=618)
    # clf = LogisticRegression(max_iter=500, n_jobs=1, random_state=618)
    clf.fit(context, treatment)
    propensity_score = clf.predict_proba(context)

    return propensity_score


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, output_dim // 2),
                                   nn.ReLU(),
                                   nn.Linear(output_dim // 2, output_dim // 2),
                                   nn.ReLU(),
                                   nn.Linear(output_dim // 2, output_dim // 2),
                                   nn.ReLU(),
                                   nn.Linear(output_dim // 2, output_dim),
                                   nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.model(x)
        return out
