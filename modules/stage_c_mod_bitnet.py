import x_transformers
from x_transformers.x_transformers import AttentionLayers, AbsolutePositionalEmbedding
from torch import nn
from modules.bitnet.attn import MultiheadAttention, MultiModalCrossAttention
from bitnet.bitffn import BitFeedForward
import torch
from torch import nn
import math


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class BitnetTransformerDecoder(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int, ff_mult=2, dropout=0., max_seq_len=64**2+1, *args, **kwargs):
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len=max_seq_len)
        
        self.layers = nn.ModuleList([])
        self.cross_layers = nn.ModuleList([])
        self.ln_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(MultiheadAttention(dim, heads, dropout=dropout))
            self.cross_layers.append(MultiModalCrossAttention(dim, heads, dim, dropout=dropout))
            self.ln_layers.append(nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)]))
            self.ffn_layers.append(BitFeedForward(dim=dim, ff_mult=ff_mult))

    def forward(self, x, context, immutable = None):
        pos_emb = self.pos_emb(x).expand(x.shape)
        x = x + pos_emb
        if immutable:
            temp = x[:, immutable, :]
        for attn, cross_attn, (ln1, ln2, ln3), ffn in zip(self.layers, self.cross_layers, self.ln_layers, self.ffn_layers):
            x = attn(x, x, x) + x
            x = ln1(x)
            x = cross_attn(x, context) + x
            x = ln2(x)
            x = ffn(x) + x
            x = ln3(x)
            if immutable:
                x[:, immutable, :] = temp
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(self.dim, self.dim)

    def forward(self, timestep: torch.Tensor):
        if len(timestep.shape) == 1:
            timestep = timestep[:, None]
        B = timestep.shape[0]
        emb = []
        for i in range(B):
            emb += [self.get_timestep_embedding(timestep[i])]
        emb = torch.cat(emb)
        return emb

    def get_timestep_embedding(self, timestep: torch.Tensor):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert len(timestep.shape) == 1

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timestep.device)
        emb = timestep.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0,1,0,0))
        emb = self.fc(emb)
        emb = nonlinearity(emb)
        return emb


class LatentEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, patch_size: int, expand_size: int = 0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.expand_size = expand_size
        self.soi = nn.Parameter(torch.randn(self.out_dim))
        self.nl = nn.Parameter(torch.randn(self.out_dim))
        self.eoi = nn.Parameter(torch.randn(self.out_dim))

        self.in_norm = nn.BatchNorm2d(num_features=self.in_dim)
        self.conv = nn.Conv2d(
            in_channels = self.in_dim,
            out_channels= self.out_dim,
            kernel_size = self.patch_size+self.expand_size*2,
            stride = self.patch_size,
            padding = self.expand_size,
            bias = False
        )
        self.out_norm = nn.BatchNorm2d(num_features=self.out_dim)
        self.temb = TimestepEmbedder(self.out_dim)
        self.ln = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B, C, H, W = x.shape
        pad_size = (
          math.ceil(H/self.patch_size)*self.patch_size-H,
          math.ceil(W/self.patch_size)*self.patch_size-W
        )
        padding = nn.ZeroPad2d((0, pad_size[1], 0, pad_size[0]))
        x = padding(x)
        B, C, H, W = x.shape
        x = self.in_norm(x)
        x = self.conv(x)
        x = self.out_norm(x)
        x = x.transpose(2, 3).transpose(1, 3)

        y = [self.temb(t), self.soi.repeat(B, 1), ]
        for i in range(H//self.patch_size):
            for j in range(W//self.patch_size):
                y += [x[:, i, j, :], ]
            y += [self.nl.repeat(B, 1), ]
        del y[-1]
        y += [self.eoi.repeat(B, 1), ]

        x = torch.cat(y, dim=1)
        x = x.reshape(B, -1, self.out_dim)
        return x
    
    
class LatentDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, patch_size):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.out_conv = nn.Sequential(
                nn.InstanceNorm2d(self.hidden_dim),
                nn.Conv2d(self.hidden_dim, self.hidden_dim*(self.patch_size**2), kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(self.patch_size),
                nn.Conv2d(self.hidden_dim, self.in_dim, kernel_size=5, padding=2, bias=False)
            )
        
    def forward(self, x: torch.Tensor, size: tuple[int, int]):
        h, w = size
        embed_h = math.ceil(h/self.patch_size)
        embed_w = math.ceil(w/self.patch_size)
        x = x[:, 2:, :]
        y = []
        for i in range(embed_h):
            y += [x[:, None, i+i*embed_w:i+(i+1)*embed_w, :]]
        y = torch.cat(y, dim=1)
        y = y.transpose(1, 3).transpose(2, 3)
        y = nonlinearity(y)
        y = self.out_conv(y)
        y = y[:, :, :h, :w]
        return y
    

class StageCTransformer(nn.Module):
    def __init__(self, 
                 in_dim: int = 16, 
                 hidden_dim: int = 1920, 
                 owl_text_dim: int = 512,
                 owl_vision_dim: int = 768,
                 patch_size: int = 2, 
                 patch_expand_size: int = 1, 
                 max_size: int = 64,
                 depth: int = 20,
                 num_heads: int = 12,
                 dropout: float = 0.,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.owl_dim = {"text": owl_text_dim, "vision": owl_vision_dim}
        self.patch_size = patch_size
        self.patch_expand_size = patch_expand_size
        self.max_size = max_size
        self.depth = depth
        self.num_heads = num_heads

        self.embedder = LatentEncoder(in_dim=self.in_dim, out_dim=self.hidden_dim, patch_size=self.patch_size, expand_size=self.patch_expand_size)
        self.dropout1d = nn.Dropout1d(dropout)
        self.decoder = BitnetTransformerDecoder(
            dim=self.hidden_dim,
            heads=self.num_heads,
            depth=self.depth,
            max_seq_len=self.max_size**2+2
        )
        self.final = LatentDecoder(self.in_dim, self.hidden_dim, self.patch_size)
        self.text_projection = nn.Linear(self.owl_dim["text"], self.hidden_dim)
        self.owl_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        
    def forward(self, x: torch.Tensor, r: torch.Tensor, text_emb: torch.Tensor):
        B, C, W, H = x.shape

        emb = self.text_projection(text_emb)
        emb = self.dropout1d(emb)
        emb = self.owl_norm(emb)
        patches = self.embedder(x, r)
        patches = self.decoder(x=patches, context=emb, immutable=0)
        x = self.final(patches, (W, H))
        return x