import torch
from torch import nn
from .activation import Gaussian, Sine

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        # out = [x]
        out = []
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 xyz_L=10, dir_L=8,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03,
                 barf_c2f=None,
                 activation=0,
                 encode_transient_front=False):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.xyz_L = xyz_L
        self.dir_L = dir_L
        self.in_channels_xyz = 6*xyz_L + 3
        self.in_channels_dir = 6*dir_L + 3

        self.encode_appearance = encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min
        self.encode_transient_front = encode_transient_front

        if activation == 0:
            ACTIVATION = nn.ReLU()
        elif activation == 1:
            ACTIVATION = Gaussian()
        elif activation == 2:
            ACTIVATION = Sine()

        # barf
        self.barf_c2f = barf_c2f
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so 

        
        add_channel = in_channels_t if self.encode_transient_front is True else 0
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz + add_channel, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, ACTIVATION)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(nn.Linear(W+self.in_channels_dir+self.in_channels_a, W//2), ACTIVATION)

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
        
        self.feat_static_layer = nn.Sequential(nn.Linear(W, W//2),
                                    ACTIVATION,
                                    nn.Linear(W//2, W//2), 
                                    ACTIVATION,
                                    nn.Linear(W//2, 384)
                                )
        add_channel = in_channels_t if self.encode_transient_front is False else 0
        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+add_channel, W//2), ACTIVATION,
                                        nn.Linear(W//2, W//2), ACTIVATION,
                                        nn.Linear(W//2, W//2), ACTIVATION,
                                        nn.Linear(W//2, W//2), ACTIVATION
                                    )
            
            self.feat_transient_layer = nn.Sequential(nn.Linear(W//2, W//2),
                                ACTIVATION,
                                nn.Linear(W//2, W//2), 
                                ACTIVATION,
                                nn.Linear(W//2, 384)
                            )
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        else:
            if output_transient:
                input_xyz, input_dir, input_a, input_t = \
                    torch.split(x, [3,
                                    3, 
                                    self.in_channels_a,
                                    self.in_channels_t], dim=-1)
            else:
                input_xyz, input_dir, input_a = \
                    torch.split(x, [3,
                                    3, 
                                    self.in_channels_a], dim=-1)
        
            input_dir = self.positional_encoding(input_dir, self.dir_L)
            input_dir_a = torch.cat([input_dir, input_a], dim=-1)
        input_xyz = self.positional_encoding(input_xyz, self.xyz_L)
        
        xyz_ = input_xyz
        if self.encode_transient_front is True:
            xyz_ = torch.cat([xyz_, input_t], 1)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        static_feat = self.feat_static_layer(xyz_encoding_final)
        
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static, static_feat
        transient_encoding_input = xyz_encoding_final
        if self.encode_transient_front is False:
            transient_encoding_input = torch.cat([transient_encoding_input, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)
        
        transient_feat = self.feat_transient_layer(transient_encoding)

        transient = torch.cat([transient_rgb, transient_sigma, transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1), torch.cat([static_feat, transient_feat], 1) 
    
    
    def positional_encoding(self, input, L): # [B,...,N]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if L == 0: return input
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=input.device)*torch.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        
        if self.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = self.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=input_enc.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(torch.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        input_enc = torch.cat([input, input_enc], -1)
        return input_enc