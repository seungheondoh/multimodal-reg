import torch
import torch.nn as nn
from einops import rearrange, repeat

from mmcr.cf_reg.modules import Transformer, JointCrossAttentionBlock
# from mmcr.constants import (NOW_CHANNEL, CREATOR)

class CFREG(nn.Module):
    def __init__(self, modality, fusion_type, head, depth, v_dim, a_dim, t_dim, g_dim, mlp_dim, output_dim, dropout, loss_fn, cls_type):
        super(CFREG, self).__init__()
        self.modality = modality
        self.fusion_type = fusion_type
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.t_dim = t_dim
        self.g_dim = g_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = loss_fn
        self.head = head
        self.mask_ratio = 0.05
        self.depth = depth
        self.cls_type = cls_type

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
        if "g" in modality:
            self.g_layer = nn.Sequential(
                nn.Linear(self.g_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.mlp_dim)
            )
            self.fusion_dim += self.mlp_dim
        
        if self.fusion_type == "mlp":
            self.fusion_layer = nn.Sequential(
                    nn.Linear(self.fusion_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.output_dim)
                )

    def projection(self, x, mode):
        x = torch.nn.functional.normalize(x, dim=1)
        x = self.dropout(x)
        if mode == "vision":
            output = self.v_layer(x)
        elif mode == "audio":
            output = self.a_layer(x)
        elif mode == "text":
            output = self.t_layer(x)
        elif mode == "genre":
            output = self.g_layer(x)
        elif mode == "fusion":
            output = self.fusion_layer(x)
        return output

    def forward(self, audio, vision, lyrics, genres, y):
        """
        x: feature embedding : vision, audio,
        """
        embs = []
        if "v" in self.modality:
            z_v = self.projection(vision, mode="vision")
            embs.append(z_v)
        if "a" in self.modality:
            z_a =self.projection(audio, mode="audio")
            embs.append(z_a)
        if "t" in self.modality:
            z_t = self.projection(lyrics, mode="text")
            embs.append(z_t)
        if "g" in self.modality:
            z_t = self.projection(genres, mode="genre")
            embs.append(z_t)        

        if self.fusion_type == "mlp":
            x = torch.cat(embs, dim=-1) 

        x = torch.nn.functional.normalize(x, dim=1)
        x = self.dropout(x)
        output = self.fusion_layer(x)
        loss = self.loss_fn(output, y) # cf item = 128 dim
        return loss
    
    def inference(self, audio, vision, lyrics, genres, y):
        """
        x: feature embedding : vision, audio,
        """
        embs = []
        if "v" in self.modality:
            z_v = self.projection(vision, mode="vision")
            embs.append(z_v)
        if "a" in self.modality:
            z_a =self.projection(audio, mode="audio")
            embs.append(z_a)
        if "t" in self.modality:
            z_t = self.projection(lyrics, mode="text")
            embs.append(z_t)
        if "g" in self.modality:
            z_t = self.projection(genres, mode="genre")
            embs.append(z_t)      

        if self.fusion_type == "mlp":
            x = torch.cat(embs, dim=-1) 
            
        x = torch.nn.functional.normalize(x, dim=1)
        x = self.dropout(x)
        output = self.fusion_layer(x)
        return output
