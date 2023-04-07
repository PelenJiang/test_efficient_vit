import math
from functools import partial
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import register_model
from timm.models.fx_features import register_notrace_module
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Attention
from timm.models.layers import SEModule

Size_ = Tuple[int, int]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        # init_channels = math.ceil(oup / ratio)
        init_channels = int(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.ghost = GhostModule(in_features, hidden_features)
        self.conv = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_features)

        self.se = False
        if self.se:
            self.senet = SEModule(hidden_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.ghost(x)
        if self.se:
            x = self.senet(x)
        x = self.conv(x)
        x = self.bn(x)

        x = x.flatten(2).transpose(1, 2)

        return x


@register_notrace_module
class LocallyGroupedAttn(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(
            B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalMILPoolingBasedAttn(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

        if self.ws > 1:
            self.mil_attention = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1)
            )

            self.propagate = nn.ConvTranspose2d(dim, dim, kernel_size=ws, stride=ws, groups=dim)
            # self.bn=nn.BatchNorm2d(dim, eps=1e-5)
            # self.act=nn.ReLU(True)
            self.norm = nn.LayerNorm(dim)

            self.proj_ = nn.Linear(dim, dim)
        else:
            self.mil_attention = None
            self.propagate = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        if self.ws > 1:
            H, W = size
            x = x.reshape(B, H, W, C)
            pad_l = pad_t = 0
            pad_r = (self.ws - W % self.ws) % self.ws
            pad_b = (self.ws - H % self.ws) % self.ws
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
            _h, _w = Hp // self.ws, Wp // self.ws
            x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
            x = x.reshape((B, _h * _w, self.ws * self.ws, C))
            A = self.mil_attention(x)
            A = torch.transpose(A, -1, -2)
            A = F.softmax(A, dim=-1)
            z = A @ x
            z = z.reshape((B, _h * _w, -1))

            qkv = self.qkv(z).reshape(B, _h * _w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, _h, _w, C)
            z = z.reshape((B, _h, _w, C))
            x = self.proj_(x) + z
            x = x.permute(0, 3, 1, 2)

            x = self.propagate(x)
            x = x.permute(0, 2, 3, 1)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            x = x.reshape(B, N, C)
            x = self.norm(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class GlobalMILBasedAttn(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

        if self.ws > 1:
            self.mil_attention = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1)
            )

            self.propagate = nn.ConvTranspose2d(dim, dim, kernel_size=ws, stride=ws, groups=dim)
            # self.bn=nn.BatchNorm2d(dim, eps=1e-5)
            # self.act=nn.ReLU(True)
            self.norm = nn.LayerNorm(dim)
        else:
            self.mil_attention = None
            self.propagate = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        if self.ws > 1:
            H, W = size
            x = x.reshape(B, H, W, C)
            pad_l = pad_t = 0
            pad_r = (self.ws - W % self.ws) % self.ws
            pad_b = (self.ws - H % self.ws) % self.ws
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
            _h, _w = Hp // self.ws, Wp // self.ws
            x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
            x = x.reshape((B, _h * _w, self.ws * self.ws, C))
            A = self.mil_attention(x)
            A = torch.transpose(A, -1, -2)
            A = F.softmax(A, dim=-1)
            z = A @ x
            z = z.reshape((B, _h * _w, -1))

            qkv = self.qkv(z).reshape(B, _h * _w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, _h, _w, C).permute(0, 3, 1, 2)
            # x = F.interpolate(x, size=(self.ws * _h, self.ws * _w))
            x = self.propagate(x)
            # x=self.bn(x)
            # x=self.act(x)
            x = x.permute(0, 2, 3, 1)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            x = x.reshape(B, N, C)
            x = self.norm(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class ConvUnit(nn.Module):
    def __init__(self, dim):
        super(ConvUnit, self).__init__()
        ratio = 2
        hidden_dim = dim * ratio
        self.conv_unit = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x, size: Size_):
        H, W = size
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.conv_unit(x) + x

        x = x.flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            self.attn = GlobalMILBasedAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GhostMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # if ws != 1:
        #     self.conv_unit = ConvUnit(dim)
        # else:
        #     self.conv_unit = None

    def forward(self, x, size: Size_):
        # if self.conv_unit:
        #     x = self.conv_unit(x, size)
        x = x + self.drop_path(self.attn(self.norm1(x), size))
        x = x + self.drop_path(self.mlp(self.norm2(x), *size))
        return x


class PosConv(nn.Module):
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim), )
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        out_size = (H // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


class EfficientVit(nn.Module):
    def __init__(
            self, img_size=224, patch_size=2, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256, 512),
            num_heads=(2, 4, 8, 16), mlp_ratios=(4, 4, 4, 4), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(2, 2, 10, 4), sr_ratios=(8, 4, 2, 1), wss=(7, 7, 7, 7),
            block_cls=Block, stem_channel=32):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]

        self.stem_conv1 = nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.ReLU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.ReLU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.ReLU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        img_size = img_size // 2
        img_size = to_2tuple(img_size)
        prev_chs = stem_channel
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.norm = norm_layer(self.num_features)

        # classification head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]

        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x.mean(dim=1)  # GAP here

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_mil_model(pretrained=False, **kwargs):
    default_cfg = _cfg()
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)

    model = EfficientVit(img_size=img_size, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfg
    return model


@register_model
def effi_vit_base(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=1, embed_dims=[64, 128, 256, 512], stem_channel=32, num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 2], wss=[6, 6, 6, 6], sr_ratios=[6, 3, 3, 1], **kwargs)
    return _create_mil_model(pretrained, **model_kwargs)


@register_model
def effi_vit_base_sr(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=1, embed_dims=[64, 128, 256, 512], stem_channel=32, num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 2], wss=[6, 3, 3, 6], sr_ratios=[6, 3, 3, 1], **kwargs)
    return _create_mil_model(pretrained, **model_kwargs)


@register_model
def effi_vit_base_5(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=1, embed_dims=[64, 128, 256, 512], stem_channel=32, num_heads=[2, 4, 8, 16], mlp_ratios=[4,4,4,4],
        depths=[2, 2, 10, 2], wss=[6, 6, 6, 6], sr_ratios=[6, 3, 3, 1], **kwargs)
    return _create_mil_model(pretrained, **model_kwargs)


@register_model
def effi_vit_base_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=2, embed_dims=[64, 128, 256, 512], stem_channel=32, num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 2], wss=[7, 7, 7, 7], sr_ratios=[7, 4, 2, 1], **kwargs)
    return _create_mil_model(pretrained, **model_kwargs)


if __name__ == '__main__':
    x = torch.randn((4, 3, 96, 96))
    model = timm.models.create_model('effi_vit_base_5', num_classes=2)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    y = model(x)
    print(1)
    # model=GlobalConvBasedAttn(dim=64,num_heads=2,ws=7)
    # y=model(x,(56,56))
    # print(y.shape)

    # model=timm.models.create_model('mil_small')
    # print(model)
    #
    # x=torch.randn((4,3,224,224))
    # y=model(x)
    # print(y)
