import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.distributions.categorical import Categorical

class HierachicalGrouping(nn.Module):
    """
        New propensity-based balancing methods
        - Pure propensity-based estimator
        - Latent treatment representation + density ratio
        - Latent treatment representation + Reconstruction loss + density ratio
    """

    def __init__(self,
                treatment_dim,
                context_dim,
                pred_hidden_size,
                device,
                embedding_size,
                bce=False):
        super(HierachicalGrouping, self).__init__()
        self.device = device
        self.treatment_dim=treatment_dim
        self.bce = bce 

        self.embedding = nn.Sequential(
            nn.Linear(treatment_dim, int(treatment_dim/2)),
            nn.ELU(),
            nn.Linear(int(treatment_dim/2), int(treatment_dim/4)),
            nn.ELU(),
            nn.Linear(int(treatment_dim/4), embedding_size),
        )

        # Prediction modules combine both the extracted z latent representation
        # and context to regress outcome Y.
        input_dim = context_dim + embedding_size
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        self.C = []
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                self.C.append(torch.max(torch.sum(torch.abs(layer.weight), axis=1)))


        self.sigmoid = nn.Sigmoid()
    def forward(self, x, t, useGroup=False):
        """
            - output for prediction.
            - z as the latent treatment.
            - reconstructed as the reconstructed t from decoder.
        """
        z = self.embedding(t)

        # print(action)
        
        # Weight normalization
        self.Softplus =  nn.Softplus()
        count = 0
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                self.C[count] = torch.max(torch.sum(torch.abs(layer.weight), axis=1))
                self.weight_normalization(layer.weight, self.Softplus(self.C[count]))
                count += 1


        concat_x = torch.cat((x, z), 1)
        output = self.predictor(concat_x)    

        if self.bce:
            output = self.sigmoid(output)

        return output, z

    
    def weight_normalization(self, W, softplus_c):
        """
            Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.Tensor([1.0]).to(self.device), softplus_c/absrowsum)
        return W * scale[:,None]

    
    def get_lipschitz_loss(self):
        """
            Compute the Lipschitz regularization loss.
        """
        loss_lip = 1.0

        for c in self.C:
            loss_lip = loss_lip * self.Softplus(c)
        return loss_lip