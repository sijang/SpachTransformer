import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.SwinTransformer import SwinTransformer

def to_3d(x):
    """
    Converts a 4D tensor to a 3D tensor by combining height, width, and depth into a single dimension.
    Args:
        x (torch.Tensor): The 4D input tensor.
    Returns:
        torch.Tensor: The reshaped 3D tensor.
    """
    return rearrange(x, 'b c h w d -> b (h w d) c')

def to_4d(x, h, w, d):
    """
    Converts a 3D tensor back to a 4D tensor by splitting the combined dimension into height, width, and depth.
    Args:
        x (torch.Tensor): The 3D input tensor.
        h (int): The height of the output tensor.
        w (int): The width of the output tensor.
        d (int): The depth of the output tensor.
    Returns:
        torch.Tensor: The reshaped 4D tensor.
    """
    return rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

class BiasFreeLayerNorm(nn.Module):
    """
    Implements a Layer Normalization module without a bias term.
    """
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBiasLayerNorm(nn.Module):
    """
    Implements a Layer Normalization module with both a bias and a scale term.
    """
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """
    Layer Normalization module that can be configured to use either bias-free or with-bias normalization.
    """
    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w, d = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), h, w, d)

class GDFN(nn.Module):
    """
    Implements a Gated-Dconv Feed-Forward Network (GDFN) module.
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.md = hidden_features * 2
        self.project_in = nn.Conv3d(dim, self.md, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(self.md, self.md, kernel_size=3, stride=1, padding=1, groups=self.md, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class GCFN(nn.Module):
    """
    Implements a Gated-Conv Feed-Forward Network (GCFN) module.
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.md = hidden_features * 2
        self.project_in = nn.Conv3d(dim, self.md, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(self.md, self.md, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    """
    Implements the Multi-DConv Head Transposed Self-Attention (MDTA) mechanism.
    """
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        m = 3
        self.qkv = nn.Conv3d(dim, dim * m, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * m, dim * m, kernel_size=3, stride=1, padding=1, groups=dim * m, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w d -> b head c (h w d)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w d) -> b (head c) h w d', head=self.num_heads, h=h, w=w, d=d)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    """
    Basic Transformer block consisting of a LayerNorm, an Attention mechanism, and a Feed Forward Network.
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.gdfn  = GDFN(dim, ffn_expansion_factor, bias)
        self.norm3 = LayerNorm(dim, layer_norm_type)
        self.gcfn = GCFN(dim, ffn_expansion_factor, bias)
        

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.gdfn(self.norm2(x))
        x = x + self.gcfn(self.norm3(x))
        
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlapped image patch embedding using a 3D Convolution.
    """
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    """
    Downsamples the input features using Conv3D and PixelUnshuffle.
    """
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    """
    Upsamples the input features using Conv3D and PixelShuffle.
    """
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class DownsampleSimple(nn.Module):
    """
    Simplified version of downsampling using Conv3D and trilinear interpolation.
    """
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        return self.body(x)

class UpsampleSimple(nn.Module):
    """
    Simplified version of upsampling using Conv3D and trilinear interpolation.
    """
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        return self.body(x)


class SpachTransformer(nn.Module):
    """
    SpachTransformer model combining Swin Transformer and Restormer architectures for image processing tasks.
    """
    def __init__(self, inp_channels=1, out_channels=1, dim=16, num_blocks=[1, 1, 1, 1],
                 heads=[1, 2, 4, 8], ffn_expansion_factor=1, bias=False,
                 layer_norm_type='WithBias', swin_window_size=(6, 6, 6)):
        super().__init__()

        # Initialize parameters and configurations
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.swin_window_size = swin_window_size

        # Encoder layers
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type 
                                     ) for _ in range(num_blocks[0])]
        )

        # Define other encoder levels and downsampling steps
        self.down1_2 = DownsampleSimple(dim)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type, 
                                     ) for _ in range(num_blocks[1])]
        )

        self.down2_3 = DownsampleSimple(int(dim*2))
        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type, 
                                     ) for _ in range(num_blocks[2])]
        )

        self.down3_4 = DownsampleSimple(int(dim*4))
        self.latent = nn.Sequential(
            *[TransformerBlock(dim=int(dim*8), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type, 
                                     ) for _ in range(num_blocks[3])]
        )

        self.reduce_chan_level4 = nn.Conv3d(int(dim*2**4), int(dim*2**3), kernel_size=1, bias=bias)

        # Decoder layers
        self.up4_3 = UpsampleSimple(int(dim * 8))
        self.reduce_chan_level3 = nn.Conv3d(int(dim * 8)+int(dim*4), int(dim * 4), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type, 
                                     ) for _ in range(num_blocks[2])]
        )

        self.up3_2 = UpsampleSimple(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv3d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type, 
                                     ) for _ in range(num_blocks[1])]
        )

        self.up2_1 = UpsampleSimple(int(dim * 2))
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type, 
                                     ) for _ in range(num_blocks[0])]
        )

        # Refinement blocks (optional)
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                                     bias=bias, layer_norm_type=layer_norm_type, 
                                     ) for _ in range(3)]  # Number of refinement blocks can be adjusted
        )

        # Final output layer
        self.output = nn.Conv3d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


        # Swin Transformer configuration 
        swin_depths      = (4, 8)
        swin_num_heads   = (4, 8)
        swin_out_indices = (0, 1)
        swin_window_size = self.swin_window_size

        # Swin Transformer instantiation
        self.swin = SwinTransformer(
            patch_size=4, in_chans=inp_channels, embed_dim=dim*2**2,
            depths=swin_depths, num_heads=swin_num_heads, window_size=swin_window_size,
            mlp_ratio=4, qkv_bias=False, drop_rate=0, drop_path_rate=0.3,
            ape=True, spe=False, patch_norm=True, use_checkpoint=False, 
            out_indices=swin_out_indices, pat_merg_rf=4, pos_embed_method='relative'
        )




    def forward(self, inp_img):
        # Process input through patch embedding and initial encoding levels
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        # Process through the latent level
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)

        # Swin Transformer processing
        swin_output = self.swin(inp_img)

        # Processing Swin Transformer outputs
        swin_latent, swin_out_enc_level1 = swin_output[-1], swin_output[-2]
        latent = torch.cat([latent, swin_latent], 1)
        latent = self.reduce_chan_level4(latent)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3, swin_out_enc_level1], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)


        # Decoder processing
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        # Final output layer
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


