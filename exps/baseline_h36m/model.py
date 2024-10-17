import copy

import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange
from GCN import GCN

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        # Rearranges a tensor from shape (b, n, d) to (b, d, n) and inverse
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc


        self.GCN_in = True
        self.GCN_out = True

        if self.temporal_fc_in:
            # if temporal_fc_in, Linear input and output with dimensions of dct matrix
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        elif self.GCN_in:
            self.motion_GCN_in = GCN(66, 66, 0.3, 1, 50)
        else:
            # if not temporal_fc_in nor GCN, Linear input and output with data dim (22*3)
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        # same with output Linear
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        elif self.GCN_out:
            self.motion_GCN_out = GCN(66, 66, 0.3, 1, 50)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        # if gcn_out not used reset motion_fc_out params
        if not self.GCN_out:
            self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights (xavier dist) and biases (init to 0) of fc_out
        """
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):

        # process motion input with Linear after dct transform
        if self.temporal_fc_in:
            # transpose d and n dims to actuate on temporal dim
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        elif self.GCN_in:
            motion_feats = self.motion_GCN_in(motion_input)
        else:
            # pass motion input directly and after that transpose to start working in temporal dimension with motion mlp
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        # compute motion_feats with motion mlp (42 layers of MLP + LN)
        motion_feats = self.motion_mlp(motion_feats)

        # process motion feats wit Linear before inverse dct
        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        elif self.GCN_out:
            motion_feats = self.motion_GCN_out(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats

