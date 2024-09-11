import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from copy import deepcopy
from collections import OrderedDict
from torch.cuda import amp
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class MeanTeacher(nn.Module):
    def __init__(self, backbone1, backbone2, args):
        super(MeanTeacher, self).__init__()
        self.pixpro_momentum = args.momentum
        self.contras_step = args.all_steps

        self.encoder = backbone1
        self.encoder.load_state_dict(torch.load(args.model_path))
        self.encoder_k = backbone2


        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # self.K = int(args.num_instances * 1. / 1 / args.batch_size * args.all_steps)
        # self.k = int(args.num_instances * 1. / 1 / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self,idx):
        """
        Momentum update of the key encoder
        """
        # _contrast_momentum = 1. - (1. - self.pixpro_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        _contrast_momentum = 1 - (1 - self.pixpro_momentum) * (math.cos(math.pi * idx / self.contras_step) + 1) / 2.
        # self.k = self.k + 1
        # print("K:",self.K,"k:",self.k)

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)


    def forward(self, idx, seismic_w, seismic_s, logCube):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        output_s, x_feat_s = self.encoder(seismic_s, logCube)  # queries: NxC


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder(idx)  # update the key encoder
            output_w, x_feat_w = self.encoder_k(seismic_w, logCube)
            # proj_2_ng = F.normalize(proj_2_ng, dim=1)

        return output_s, x_feat_s, output_w, x_feat_w
