import torch
import torch.nn as nn


class RDLoss(nn.Module):
    def __init__(self, ):
        super(RDLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, C, K, R1, R2, R3, R4, R5, D1, D2, D3, D4, D5):

        pred_D1 = C + K * torch.log(R1)
        pred_D2 = C + K * torch.log(R2)
        pred_D3 = C + K * torch.log(R3)
        pred_D4 = C + K * torch.log(R4)
        pred_D5 = C + K * torch.log(R5)

        return self.l1_loss(pred_D1, D1) + self.l1_loss(pred_D2, D2) + self.l1_loss(pred_D3, D3) + self.l1_loss(pred_D4, D4) + self.l1_loss(pred_D5, D5)