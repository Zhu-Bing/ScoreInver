import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp


class lambda_loss(nn.Module):
    def __init__(self):
        super(lambda_loss, self).__init__()

    @amp.autocast()
    def forward(self, y, target, flag):
        loss = None
        # print(torch.where(target == -999.))
        mask = torch.where(target == -999., 0., 1.)
        if flag == 'bce':
            loss = torch.sum(F.binary_cross_entropy_with_logits(y, target, reduce=False) * mask) / torch.sum(mask)
        if flag == 'l1':
            loss = torch.sum(F.l1_loss(y, target, reduce=False) * mask) / torch.sum(mask)

        return loss
