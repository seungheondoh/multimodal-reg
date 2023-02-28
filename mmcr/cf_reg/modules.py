import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
    
    def loss(self, h1, h2):
        device = h1.device
        logits = torch.einsum('nc,mc->nm', [h1, h2])
        logits /= self.temperature
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def forward(self, h1, h2):        
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        loss = (self.loss(h1, h2) + self.loss(h2, h1)) / 2
        return loss

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): 
        """
        x - > b x l x d
        """
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') 
        return self.to_out(out)



class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 1, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)


    def forward(self, x): 
        # b x 4 x 128
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b x 4 x 4 
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) 
        out = rearrange(out, 'b h n d -> b n (h d)') # multi-head flatten
        # out = self.to_out(out) # weighted sum - linear module
        return attn, out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True)
    return t.softmax(dim = dim)

# bidirectional cross attention - have two sequences attend to each other with 1 attention step
class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self, dim,heads = 8, dim_head = 64, context_dim = None, dropout = 0.,talking_heads = False, prenorm =True,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = False
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device
        x = self.norm(x)
        context = self.context_norm(context)
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)
        # split out head
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # get similarities
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale
        
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = stable_softmax(sim, dim = -1)
        context_attn = stable_softmax(sim, dim = -2)

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        
        # src sequence aggregates values from context, context aggregates values from src sequence
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)
        
        # merge heads and combine out
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)
        return out, context_out


class JointCrossAttentionBlock(nn.Module):
    def __init__(
        self, dim, context_dim, depth, mlp_dim, dropout = 0.
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.cross_attns = nn.ModuleList([])
        for _ in range(depth):
            self.cross_attns.append(nn.ModuleList([
                    BidirectionalCrossAttention(dim = dim, context_dim = context_dim, dropout = dropout, prenorm = True),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                    FeedForward(context_dim, mlp_dim, dropout=dropout)
                    ]
                )
            )

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None
    ):
        for attn, ff, context_ff in self.cross_attns:
            attn_out, context_attn_out = attn(x, context, mask = mask, context_mask = context_mask)
            x = x + attn_out # residual
            context = context + context_attn_out # residual
            x = ff(x) + x # residual
            context = context_ff(context) + context # residual
        return x, context