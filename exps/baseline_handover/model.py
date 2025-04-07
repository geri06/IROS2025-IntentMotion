import copy

import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange
from GCN import GCN
import csv

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

        self.gcn_in = config.motion_gcn_in.gcn_in
        self.gcn_out = config.motion_gcn_out.gcn_out

        # Add REE conditioning
        self.ree_cond = config.motion_ree.ree_cond
        self.ree_concatenation = config.motion_ree.ree_concatenation
        self.GCN_concatenation = config.motion_ree.gcn_concatenation

        # Add Int conditioning
        self.int_cond = config.motion_int.int_cond

        # Add Int classifier
        self.use_int_class = config.use_int_class
        self.pred_dim = config.motion.handover_target_length_train
        self.flatten = config.classifier.flatten

        if self.temporal_fc_in:
            # if temporal_fc_in, Linear input and output with dimensions of dct matrix
            self.motion_fc_in = nn.Linear(self.config.motion.handover_input_length_dct, self.config.motion.handover_input_length_dct)
        elif self.gcn_in:
            self.motion_gcn_in = GCN(self.config.motion_gcn_in.in_features, self.config.motion_gcn_in.out_features, self.config.motion_gcn_in.do, self.config.motion_gcn_in.num_stage , self.config.motion_gcn_in.n_node)
        else:
            # if not temporal_fc_in nor GCN, Linear input and output with data dim (22*3)
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        # same with output Linear
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.handover_input_length_dct, self.config.motion.handover_input_length_dct)
        elif self.gcn_out:
            self.motion_gcn_out = GCN(self.config.motion_gcn_out.in_features, self.config.motion_gcn_out.out_features, self.config.motion_gcn_out.do, self.config.motion_gcn_out.num_stage , self.config.motion_gcn_out.n_node)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        # if gcn_out not used reset motion_fc_out params
        if not self.gcn_out:
            self.reset_parameters()

        # initialize ree networks
        if self.ree_cond:
            self.motion_ree = nn.Linear(self.config.motion_ree.input_dim, self.config.motion_ree.embedding_size)

            # initialize network to reduce dim of ree and motion input concatenation (from N to 27)
            if self.ree_concatenation:
                if self.GCN_concatenation:
                    self.motion_context = GCN(self.config.motion_ree.embedding_size + self.config.motion.dim, self.config.motion.dim,
                        self.config.motion_ree.gcn_do, self.config.motion_ree.gcn_num_stage,
                        self.config.motion_ree.gcn_n_node)
                else:
                    self.motion_context = nn.Linear(self.config.motion_ree.embedding_size + self.config.motion.dim, self.config.motion.dim)

        # initialize intention network
        if self.int_cond:
            self.motion_int = nn.Embedding(self.config.motion_int.num_emb, self.config.motion_int.output_dim)

        if self.use_int_class:
            if self.flatten:
                input_dim = self.config.motion.dim * self.config.motion.handover_target_length_train
            else:
                input_dim = self.config.motion.dim
            self.int_classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),  # Activation
                nn.Linear(128, 64),
                nn.ReLU(),  # Activation
                nn.Linear(64, self.config.motion_int.num_emb)  # Output to number of intention classes
            )

    def reset_parameters(self):
        """
        Initializes weights (xavier dist) and biases (init to 0) of fc_out
        """
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, ree_input, int_input):

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # process motion input without context with Linear after dct transform
        if self.temporal_fc_in:
            # transpose d and n dims to actuate on temporal dim
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        elif self.gcn_in:
            motion_feats = self.motion_gcn_in(motion_input)
            motion_feats = self.arr0(motion_feats)
        else:
            # pass motion input directly and after that transpose to start working in temporal dimension with motion mlp
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        # If we want to add ree context process it through motion_ree layer and concat
        if self.ree_cond:
            ree_feats = self.motion_ree(ree_input)
            ree_feats = ree_feats.unsqueeze(1).repeat(1, self.config.motion.handover_input_length_dct, 1)
            motion_feats = self.arr1(motion_feats)
            # ree_feats = self.arr0(ree_feats)

            if self.ree_concatenation:
                context_feats = torch.cat((motion_feats, ree_feats), dim=2)
                motion_feats = self.motion_context(context_feats)
                motion_feats = self.arr0(motion_feats)
            else:
                # add context by adding values to motion input with dim 27
                motion_feats += ree_feats
                motion_feats = self.arr0(motion_feats)

        if self.int_cond:
            int_input = int_input.int()
            int_feats = self.motion_int(int_input)
            int_feats = int_feats.unsqueeze(1).repeat(1,self.config.motion.handover_input_length_dct,1)
            int_feats = self.arr1(int_feats)
            motion_feats += int_feats

        # compute motion_feats with motion mlp (42 layers of MLP + LN)
        motion_feats = self.motion_mlp(motion_feats)

        # if self.int_cond:
        #     # we add again intention embedding
        #     motion_feats += int_feats

        # process motion feats wit Linear before inverse dct
        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        elif self.gcn_out:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_gcn_out(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        if self.use_int_class:
            #print(torch.flatten(motion_feats[:,:self.pred_dim,:],start_dim=1).shape)
            if self.flatten:
                int_class_logits = self.int_classifier(torch.flatten(motion_feats[:,:self.pred_dim,:],start_dim=1))  # Flatten dims 1 and 2 considering the 10 first predited frames
            else:

                int_class_logits = self.int_classifier(motion_feats[:,:self.pred_dim,:].mean(dim=1))  # Pooling along temporal dimension
            int_predictions = torch.argmax(int_class_logits, dim=1)
            end_event.record()

            # Wait for everything to finish
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
            print("Elapsed Time:", elapsed_time)

            csv_file = "times.csv"

            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([elapsed_time])
        else:
            int_class_logits = torch.empty(0)
            int_predictions = torch.empty(0)
            end_event.record()
            torch.cuda.synchronize()

        return motion_feats, int_class_logits, int_predictions

