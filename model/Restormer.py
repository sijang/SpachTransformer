import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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

class FeedForward(nn.Module):
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
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
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


class Restormer(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, dim=32, num_blocks=[2,2,2,6], 
                 num_refinement_blocks=1, heads=[1,1,1,1], ffn_expansion_factor=2.66, bias=False, 
                 layer_norm_type='WithBias'):
        super(Restormer, self).__init__()

        # Initialize Patch Embedding
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder Layers
        self.encoder_layers = nn.ModuleList()
        for i in range(4):
            layer_dim = dim // (2 ** i)
            self.encoder_layers.append(nn.Sequential(*[
                TransformerBlock(dim=layer_dim, num_heads=heads[i], ffn_expansion_factor=ffn_expansion_factor, 
                                 bias=bias, layer_norm_type=layer_norm_type) 
                for _ in range(num_blocks[i])
            ]))

        # Downsampling Layers
        self.downsample_layers = nn.ModuleList([Downsample(dim // (2 ** i)) for i in range(3)])

        # Latent Layer
        self.latent = self.encoder_layers[-1]

        # Upsampling and Decoder Layers
        self.upsample_layers = nn.ModuleList([Upsample(dim // (2 ** (i + 1))) for i in range(3)])
        self.decoder_layers = nn.ModuleList()
        for i in range(3, 0, -1):
            layer_dim = dim // (2 ** i)
            self.decoder_layers.append(nn.Sequential(*[
                TransformerBlock(dim=layer_dim, num_heads=heads[i-1], ffn_expansion_factor=ffn_expansion_factor, 
                                 bias=bias, layer_norm_type=layer_norm_type) 
                for _ in range(num_blocks[i-1])
            ]))

        # Channel Reduction Layers
        self.reduce_chan_layers = nn.ModuleList([
            nn.Conv3d(dim // (2 ** i), dim // (2 ** (i + 1)), kernel_size=1, bias=bias) for i in range(1, 3)
        ])

        # Refinement Layer
        self.refinement = nn.Sequential(*[TransformerBlock(dim=dim*2, num_heads=heads[0], 
                                                           ffn_expansion_factor=ffn_expansion_factor, 
                                                           bias=bias, layer_norm_type=layer_norm_type) 
                                          for _ in range(num_refinement_blocks)])


        # Output Layer
        self.output = nn.Conv3d(dim*2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        x = self.patch_embed(inp_img)

        # Encoder
        skip_connections = []
        for i in range(4):
            x = self.encoder_layers[i](x)
            if i < 3:
                skip_connections.append(x)
                x = self.downsample_layers[i](x)

        # Latent
        latent = self.latent(x)

        # Decoder
        for i in range(3):
            x = self.upsample_layers[i](latent)
            x = torch.cat([x, skip_connections[2-i]], 1)
            if i < 2:
                x = self.reduce_chan_layers[i](x)
            x = self.decoder_layers[i](x)

        # Refinement
        x = self.refinement(x)

        # Dual-Pixel Task
        if self.dual_pixel_task:
            x = x + self.skip_conv(skip_connections[0])
            x = self.output(x)
        else:
            x = self.output(x) + inp_img

        return x
