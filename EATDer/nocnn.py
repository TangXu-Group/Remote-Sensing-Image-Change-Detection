
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from CBAMnet import *


class NLBlockND_cross(nn.Module):


    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=2, bn_layer=True):

        super(NLBlockND_cross, self).__init__()

        

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:#防止输入错误
                self.inter_channels = 1


        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d
        

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)


        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):

        batch_size = x_thisBranch.size(0)


        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)#完成nonlocal的乘积

     
        else:
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) 
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
 
        z = W_y + x_thisBranch

        return z





class convBlock(nn.Module):
  
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
   
    def __init__(self, 
                 depths=[ 2, 2], dims=[128,160, 192], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() 
        
        for i in range(2):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(2):
            stage = nn.Sequential(
                *[convBlock(dim=dims[i+1], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) 
        
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        for i in range(2):
            x = self.downsample_layers[i](x)
            
            x = self.stages[i](x)
            
        return x

class LayerNorm(nn.Module):
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False,pool=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        self.act = nn.GELU()
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(pool)#自适应平均池化固定输出nxn
            
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = self.act(x)
                kv = self.kv(x_ ).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False,pool=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear,pool=pool)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x


class OverlapPatchEmbed(nn.Module):
   

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=512):
        super().__init__()
        
        
        patch_size = to_2tuple(patch_size)
        
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        
        
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W



class SAVT_refine(nn.Module):
    def __init__(self,embed_dims=256):#
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.blk1=Block(
            dim=embed_dims, num_heads=1, mlp_ratio=8, qkv_bias=True, qk_scale=None,
            drop=0, attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm,
            sr_ratio=8, linear=True) 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        
        B, _, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
       
        x = self.blk1(x, H, W)

        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x

class SAVT_refine2(nn.Module):
    def __init__(self,embed_dims=42):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        self.blk1=Block(
            dim=embed_dims, num_heads=1, mlp_ratio=8, qkv_bias=True, qk_scale=None,
            drop=0, attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm,
            sr_ratio=8, linear=True) 
  
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        
        B, _, H, W = x.shape#这里记录的hw是本block的PATCH大小
        
        x = x.flatten(2).transpose(1, 2)#将hw压缩为一个维度之后转换两个维度的位置 这就是4096的来源
        x = self.norm(x)
        
       
        x = self.blk1(x, H, W)
        #x = self.blk2(x, H, W)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return x



def sqrt_and_ceil(lst):
    result = []
    for num in lst:
        result.append(math.ceil(math.sqrt(num)))
    return result


class SAVT_CD(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4,3], sr_ratios=[8, 4], num_stages=3, linear=False,stage_imgscale=[64,32,16]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        
        self.crossnet11=NLBlockND_cross(64)
        self.crossnet12=NLBlockND_cross(64)
        self.crossnet21=NLBlockND_cross(128)
        self.crossnet22=NLBlockND_cross(128)
        self.crossnet31=NLBlockND_cross(128)
        self.crossnet32=NLBlockND_cross(128)
        
        poollist=sqrt_and_ceil(stage_imgscale)
       
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear,pool=poollist[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features1(self, x):
        B = x.shape[0]
        
        i=0
            
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
       
        for blk in block:
            x = blk(x, H, W)
            
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x
    
    def forward_features2(self, x):
        B = x.shape[0]
        
        i=1
            
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        
        for blk in block:
            x = blk(x, H, W)
            
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x
    
    def forward_features3(self, x):
        B = x.shape[0]
        
        i=2
            
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        
        for blk in block:
            x = blk(x, H, W)
            
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x,y):
        x_1 = self.forward_features1(x)
        y_1 = self.forward_features1(y)
        
        x1 = self.crossnet11(x_1,y_1)
        y1 = self.crossnet12(y_1,x_1)
        
        x_2 = self.forward_features2(x1)
        y_2 = self.forward_features2(y1)
        
        x2 = self.crossnet21(x_2,y_2)
        y2 = self.crossnet22(y_2,x_2)
        
        
        x_3 = self.forward_features3(x2)
        y_3 = self.forward_features3(y2)
        
        x3 = self.crossnet31(x_3,y_3)
        y3 = self.crossnet32(y_3,x_3)
       
       
        return x1,y1,x2,y2,x3,y3


    
    
class uplayersCF(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(uplayersCF, self).__init__()        
        
        self.cov= nn.Sequential(
            
            nn.Conv2d(inchannel*2, outchannel, 3, stride=1, padding=1), 
            nn.GELU(),
            nn.BatchNorm2d(outchannel),
            
            )
    def forward(self, x1,x2):

        x3=torch.cat((x1,x2),dim=1)
        x3=self.cov(x3)
        x3=nn.UpsamplingBilinear2d(size=(64,64))(x3)
        return x3


class uplayersCF2(nn.Module):
     def __init__(self,inchannel):
         super(uplayersCF2, self).__init__()        
         
         self.cov= nn.Sequential(
             
             nn.Conv2d(inchannel*2, inchannel, 3, stride=1, padding=1), 
             nn.GELU(),
             nn.BatchNorm2d(inchannel),
             nn.ConvTranspose2d(inchannel,int(inchannel/2),kernel_size=2, stride=2),
             nn.GELU(),
             nn.BatchNorm2d(int(inchannel/2)),
             nn.ConvTranspose2d(int(inchannel/2),int(inchannel/2),kernel_size=2, stride=2),
             nn.GELU(),
             nn.BatchNorm2d(int(inchannel/2)),
             )
     def forward(self, x1,x2):

         x3=torch.cat((x1,x2),dim=1)
         x3=self.cov(x3)
         x3=nn.UpsamplingBilinear2d(size=(64,64))(x3)
         
         return x3   
    


class aftertransCF2(nn.Module):
    def __init__(self, embed_dims):
        super(aftertransCF2, self).__init__()
        self.uplayersCF1=uplayersCF(embed_dims[0],embed_dims[0]*2)
        self.uplayersCF2=uplayersCF(embed_dims[1],embed_dims[1]*2)
        self.uplayersCF3=uplayersCF(embed_dims[1],embed_dims[0])
        self.CAM1=CBAM(embed_dims[0]*2)
        self.CAM2=CBAM(embed_dims[1]*2)
        self.CAM3=CBAM(embed_dims[0])
    def forward(self, a,b,c,d,e,f):
        
        
        x1=self.uplayersCF1(a,b)
        x2=self.uplayersCF2(c,d)
        
        x3=self.uplayersCF3(e,f)
        x1=self.CAM1(x1)
        x2=self.CAM2(x2)
        x3=self.CAM3(x3)
        
        return x1,x2,x3
    

class aftertransCF21(nn.Module):
    def __init__(self, embed_dims):
        super(aftertransCF21, self).__init__()
        
        self.conv= nn.Sequential(nn.Conv2d(embed_dims[1]*2+embed_dims[0]*3, embed_dims[1]*2,3, stride=1, padding=1),
                                 nn.BatchNorm2d(embed_dims[1]*2),
                                 nn.GELU())
        
        self.unconv=nn.Sequential(nn.ConvTranspose2d(embed_dims[1]*2,int(embed_dims[1]),kernel_size=2, stride=2),
                                  nn.BatchNorm2d(int(embed_dims[1])),
                                  nn.GELU(),nn.ConvTranspose2d(int(embed_dims[1]),int(embed_dims[1]/3),kernel_size=2, stride=2),
                                  nn.BatchNorm2d(int(embed_dims[1]/3)),
                                  nn.GELU())

       
        
        self.conv2=nn.Sequential(nn.Conv2d( int(embed_dims[1]/3),int(embed_dims[1]/4),3,1,1),
                                  nn.GELU(),
                                  nn.BatchNorm2d(int(embed_dims[1]/4)),
                                  nn.Conv2d( int(embed_dims[1]/4),int(embed_dims[1]/6),3,1,1),
                                  nn.GELU(),
                                  nn.BatchNorm2d(int(embed_dims[1]/6)),
                                  nn.Conv2d( int(embed_dims[1]/6),int(embed_dims[1]/10),1,1),
                                  nn.GELU(),
                                  nn.BatchNorm2d(int(embed_dims[1]/10)),
                                  nn.Conv2d( int(embed_dims[1]/10),2,1,1)
                                  )
        
        
        self.refine=SAVT_refine()
        self.refine2=SAVT_refine2()
        
        
    def forward(self, x1,x2,x3):
        
        x=torch.cat((x1,x2,x3),dim=1)  
        
        x=self.conv(x)
        x=self.refine(x)
        x=self.unconv(x)
        x=self.refine2(x)
        x=self.conv2(x)
    
        edge=x[:,0].unsqueeze(1)
        
        block=x[:,1].unsqueeze(1)
        return edge,block




class EATDer(nn.Module):
    def __init__(self):
        super(EATDer, self).__init__()
        self.net1=SAVT_CD(img_size=256,patch_size=4, num_heads=[4, 4, 4, 4], 
                                   mlp_ratios=[8, 8, 4, 4], qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   depths=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1], linear=True,)
        self.net2=aftertransCF2(embed_dims=[64, 128, 512, 512])
        self.net3=aftertransCF21(embed_dims=[64, 128,512])
        
        
        
        
        
    def forward(self, a,b):

        x1,y1,x2,y2,x3,y3= self.net1(a,b)#x2 y2是savt的cross输出
        
        x_1,x_2,x_3=self.net2(x1,y1,x2,y2,x3,y3)
        
        edge,block=self.net3(x_1,x_2,x_3)
        
        
        return edge,block


class DWConv(nn.Module):
    def __init__(self, dim=512):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x







from thop import profile
from thop import clever_format
testmodel=EATDer()
#print(summary(testmodel, input_size=[(1,3, 256,256), (1,3, 256,256)]))
dummy_input = torch.randn((1,3, 256,256))
flops, params = profile(testmodel, (dummy_input,dummy_input))
flops, params = clever_format([flops, params], '%.3f')
print('flops: ', flops, 'params: ', params)