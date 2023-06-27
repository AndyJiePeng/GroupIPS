import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from tqdm import tqdm
import numpy as np
import copy
import torch.nn.functional as F
from model.HierachicalGroup import HierachicalGrouping
# Set seed 
torch.manual_seed(618)

def run_latent_method(train_x,
                    train_t,
                    train_y,
                    group_k,
                    ori_k,
                    params,
                    device,
                    embedding_size,
                    BCE=False
                    ):
    lr = params['lr']
    batch_size = params['batch_size']
    epochs = params['epochs']
    train_x = torch.from_numpy(train_x).float()
    onehot_t = F.one_hot(torch.from_numpy(train_t).long().squeeze())
    # onehot_t = torch.from_numpy(train_t).float()
    train_y = torch.from_numpy(train_y).float()
    labelTrain_t = torch.from_numpy(train_t).long()    

    if BCE:
        pred_criterion = torch.nn.BCELoss()
        train_y = torch.unsqueeze(train_y, 1)
    else:
        pred_criterion = torch.nn.MSELoss()

    dataset = TensorDataset(train_x,  onehot_t, train_y, labelTrain_t)
    dataLoader = DataLoader(dataset, batch_size, shuffle=True)
    
    model = HierachicalGrouping(treatment_dim=ori_k,
                                device=device,
                                context_dim=train_x.shape[1],
                                pred_hidden_size=train_x.shape[1] + 1,
                                embedding_size=embedding_size,
                                bce=BCE).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    group_criterion = nn.CrossEntropyLoss()
    train_x = train_x.to(device)
    onehot_t = onehot_t.to(device)
    train_y = train_y.to(device)
    softmax = nn.Softmax(dim=1)
    for i in range(epochs):
        model.train()
        for x, t, y, label_t in dataLoader:
            x = x.to(device)
            label_t = label_t.squeeze().to(device)
            t =  t.float().to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x, t, useGroup=True)
            prediction, z = output 

            pred_loss = pred_criterion(prediction, y)
            lipschitz_loss = model.get_lipschitz_loss()
            # print("prediction_loss:", pred_loss)
            # print("Lipchitz_loss", lipschitz_loss)
            # print(pred_loss, lipschitz_loss)
            loss = pred_loss + 1e-3 * lipschitz_loss
            loss.backward() 


            optimizer.step()
        if (i + 1) % 50 == 0:
            # print("Variance:", var_loss)

            prediction, z = model(train_x, onehot_t.float())
            print_loss = pred_criterion(prediction, train_y)


            if i == (epochs - 1):
                one_hot =  F.one_hot(torch.Tensor([[i] for i in range(ori_k)]).long().squeeze()).float()
                _, z = model(train_x[0:ori_k].to(device), one_hot.to(device))
                print("Embedding output:", z)


            print("Epoch {}: Training mse: {}".format(i, print_loss))

    return model, onehot_t


