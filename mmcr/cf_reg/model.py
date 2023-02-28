import torch
import torch.nn as nn
from einops import rearrange, repeat

from mmcr.cf_reg.modules import Transformer, JointCrossAttentionBlock
from mmcr.constants import (NOW_CHANNEL, CREATOR)

class CFREG(nn.Module):
    def __init__(self, modality, fusion_type, head, depth, v_dim, a_dim, t_dim, mlp_dim, output_dim, dropout, loss_fn, cls_type):
        super(CFREG, self).__init__()
        self.modality = modality
        self.fusion_type = fusion_type
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.t_dim = t_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = loss_fn
        self.head = head
        self.mask_ratio = 0.05
        self.depth = depth
        self.cls_type = cls_type
        if cls_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, self.mlp_dim))
        elif cls_type == "channel":
            self.cls_token = nn.Parameter(torch.randn(len(NOW_CHANNEL) + 1, self.mlp_dim)) # add UNK token
        elif cls_type == "creator":
            self.cls_token = nn.Parameter(torch.randn(len(CREATOR) + 1, self.mlp_dim)) # add UNK token
        elif cls_type == "both":
            self.cls_token = nn.Parameter(torch.randn(len(NOW_CHANNEL) + len(CREATOR) + 1, self.mlp_dim)) # add UNK token

        self.fusion_dim = 0
        if "v" in modality:
            self.v_layer = nn.Sequential(
                nn.Linear(self.v_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.mlp_dim)
                )
            self.fusion_dim += self.mlp_dim
        if "a" in modality:
            self.a_layer = nn.Sequential(
                nn.Linear(self.a_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.mlp_dim)
            )
            self.fusion_dim += self.mlp_dim
        if "t" in modality:
            self.t_layer = nn.Sequential(
                nn.Linear(self.t_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.mlp_dim)
            )
            self.fusion_dim += self.mlp_dim
        
        if self.fusion_type == "mlp":
            self.fusion_layer = nn.Sequential(
                    nn.Linear(self.fusion_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.output_dim)
                )
        elif self.fusion_type == "transformer":
            self.self_attn = Transformer(
                dim = self.mlp_dim,
                depth = self.depth,
                heads = self.head,
                dim_head = 64,
                mlp_dim = self.mlp_dim,
                dropout = dropout
            )
            if self.cls_type == "none":
                feature_dim = self.mlp_dim * (len(self.modality))
            elif self.cls_type == "both":
                feature_dim = self.mlp_dim * (len(self.modality) + 2) # channel, creator
            else:
                feature_dim = self.mlp_dim * (len(self.modality) + 1)

            self.fusion_layer = nn.Sequential(
                    nn.Linear(feature_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.output_dim)
                )
        elif self.fusion_type == "c_transformer":
            self.trans1 = JointCrossAttentionBlock(
                dim = self.mlp_dim,
                context_dim = self.mlp_dim,
                depth = 3,
                mlp_dim = self.mlp_dim,
                dropout = dropout
            )
            self.trans2 = JointCrossAttentionBlock(
                dim = self.mlp_dim,
                context_dim = self.mlp_dim,
                depth = 3,
                mlp_dim = self.mlp_dim,
                dropout = dropout
            )
            self.trans3 = JointCrossAttentionBlock(
                dim = self.mlp_dim,
                context_dim = self.mlp_dim,
                depth = 3,
                mlp_dim = self.mlp_dim,
                dropout = dropout
            )
            self.trans4 = JointCrossAttentionBlock(
                dim = self.mlp_dim,
                context_dim = self.mlp_dim,
                depth = 3,
                mlp_dim = self.mlp_dim,
                dropout = dropout
            )
            feature_dim = self.mlp_dim * (len(self.modality) + 2) # channel, creator
            self.fusion_layer = nn.Sequential(
                    nn.Linear(feature_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.output_dim)
            )

    def prob_mask_like(self, t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    def projection(self, x, mode):
        x = torch.nn.functional.normalize(x, dim=1)
        x = self.dropout(x)
        if mode == "vision":
            output = self.v_layer(x)
        elif mode == "audio":
            output = self.a_layer(x)
        elif mode == "text":
            output = self.t_layer(x)
        elif mode == "fusion":
            output = self.fusion_layer(x)
        return output

    def forward(self, x_v, x_a, x_t, y, meta_id=None):
        """
        x: feature embedding : vision, audio,
        """
        embs = []
        if "v" in self.modality:
            z_v = self.projection(x_v, mode="vision")
            embs.append(z_v)
        if "a" in self.modality:
            z_a =self.projection(x_a, mode="audio")
            embs.append(z_a)
        if "t" in self.modality:
            z_t = self.projection(x_t, mode="text")
            embs.append(z_t)
        

        if self.fusion_type == "mlp":
            x = torch.cat(embs, dim=-1) 

        elif self.fusion_type == "transformer":
            x = torch.stack(embs).transpose(1,0)
            if self.cls_type == "cls":
                cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_type == "channel":
                cls_token = self.cls_token[meta_id]
                cls_token = cls_token.unsqueeze(1) 
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_type == "creator":
                creator_mask = self.prob_mask_like(meta_id, self.mask_ratio)
                creator_ids = meta_id.masked_fill(creator_mask, 0)
                cls_token = self.cls_token[creator_ids]
                cls_token = cls_token.unsqueeze(1) 
                x = torch.cat((cls_token, x), dim=1)

            elif self.cls_type == "both":
                channel_ids = meta_id[:,0]
                creator_ids = meta_id[:,1]
                creator_mask = self.prob_mask_like(creator_ids, self.mask_ratio)
                creator_ids = creator_ids.masked_fill(creator_mask, 0)
                channel_token = self.cls_token[channel_ids].unsqueeze(1) 
                creator_token = self.cls_token[creator_ids].unsqueeze(1) 
                x = torch.cat((channel_token, creator_token, x), dim=1)
            x = self.dropout(x)
            x = self.self_attn(x)
            x = rearrange(x, 'b n d -> b (n d)') 

        elif self.fusion_type == "c_transformer":
            channel_ids = meta_id[:,0]
            creator_ids = meta_id[:,1]
            creator_mask = self.prob_mask_like(creator_ids, self.mask_ratio)
            creator_ids = creator_ids.masked_fill(creator_mask, 0)
            channel_token = self.cls_token[channel_ids].unsqueeze(1) 
            creator_token = self.cls_token[creator_ids].unsqueeze(1) 
            z_v = z_v.unsqueeze(1)
            z_a = z_a.unsqueeze(1)
            z_t = z_t.unsqueeze(1)

            z_v,z_a = self.trans1(z_v,z_a)
            z_va = torch.cat((z_v, z_a), dim=1)
            z_va, z_t = self.trans2(z_va, z_t)
            z_vat = torch.cat((z_va, z_t), dim=1)
            z_vat, creator_token = self.trans3(z_vat, creator_token)
            z_vatc = torch.cat((z_vat, creator_token), dim=1)
            z_vatc, channel_token = self.trans4(z_vatc, channel_token)
            z_vatcc = torch.cat((z_vatc, channel_token), dim=1)
            x = rearrange(z_vatcc, 'b n d -> b (n d)') 
        
        x = torch.nn.functional.normalize(x, dim=1)
        x = self.dropout(x)
        output = self.fusion_layer(x)
        loss = self.loss_fn(output, y) # cf item = 128 dim
        return loss
    
    def inference(self, x_v, x_a, x_t, meta_id=None):
        """
        x: feature embedding : vision, audio,
        """
        embs = []
        if "v" in self.modality:
            z_v = self.projection(x_v, mode="vision")
            embs.append(z_v)
        if "a" in self.modality:
            z_a =self.projection(x_a, mode="audio")
            embs.append(z_a)
        if "t" in self.modality:
            z_t = self.projection(x_t, mode="text")
            embs.append(z_t)

        if self.fusion_type == "mlp":
            x = torch.cat(embs, dim=-1) 
            
        elif self.fusion_type == "transformer":
            x = torch.stack(embs).transpose(1,0)
            if self.cls_type == "cls":
                cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_type == "channel" or self.cls_type == "creator":
                cls_token = self.cls_token[meta_id]
                cls_token = cls_token.unsqueeze(1) 
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_type == "both":
                channel_token = self.cls_token[meta_id[:,0]].unsqueeze(1) 
                creator_token = self.cls_token[meta_id[:,1]].unsqueeze(1) 
                x = torch.cat((channel_token, creator_token, x), dim=1)
            x = self.dropout(x)
            x = self.self_attn(x)
            x = rearrange(x, 'b n d -> b (n d)') 

        elif self.fusion_type == "transformer":
            x = torch.stack(embs).transpose(1,0)
            if self.cls_type == "cls":
                cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_type == "channel" or self.cls_type == "creator":
                cls_token = self.cls_token[meta_id]
                cls_token = cls_token.unsqueeze(1) 
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_type == "both":
                channel_ids = meta_id[:,0]
                creator_ids = meta_id[:,1]
                channel_mask = self.prob_mask_like(channel_ids)
                creator_mask = self.prob_mask_like(creator_ids)
                channel_ids = channel_ids.masked_fill(channel_mask, 0)
                creator_ids = creator_ids.masked_fill(creator_mask, 0)
                channel_token = self.cls_token[channel_ids].unsqueeze(1) 
                creator_token = self.cls_token[creator_ids].unsqueeze(1) 
                x = torch.cat((channel_token, creator_token, x), dim=1)
            x = self.dropout(x)
            x = self.self_attn(x)
            x = rearrange(x, 'b n d -> b (n d)') 
            
        elif self.fusion_type == "c_transformer":
            channel_ids = meta_id[:,0]
            creator_ids = meta_id[:,1]
            creator_mask = self.prob_mask_like(creator_ids, self.mask_ratio)
            creator_ids = creator_ids.masked_fill(creator_mask, 0)
            channel_token = self.cls_token[channel_ids].unsqueeze(1) 
            creator_token = self.cls_token[creator_ids].unsqueeze(1) 
            z_v = z_v.unsqueeze(1)
            z_a = z_a.unsqueeze(1)
            z_t = z_t.unsqueeze(1)

            z_v,z_a = self.trans1(z_v,z_a)
            z_va = torch.cat((z_v, z_a), dim=1)
            z_va, z_t = self.trans2(z_va, z_t)
            z_vat = torch.cat((z_va, z_t), dim=1)
            z_vat, creator_token = self.trans3(z_vat, creator_token)
            z_vatc = torch.cat((z_vat, creator_token), dim=1)
            z_vatc, channel_token = self.trans4(z_vatc, channel_token)
            z_vatcc = torch.cat((z_vatc, channel_token), dim=1)
            x = rearrange(z_vatcc, 'b n d -> b (n d)') 
        
        x = torch.nn.functional.normalize(x, dim=1)
        x = self.dropout(x)
        output = self.fusion_layer(x)
        return output
