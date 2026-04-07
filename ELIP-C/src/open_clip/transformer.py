import logging
from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
from functools import partial

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import T
from torch.utils.checkpoint import checkpoint

from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

import ipdb



class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            # print('self.ln_post', x.shape)
            x = self.ln_post(x)
            # print('self._global_pool', x.shape)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            # print('self.proj', pooled.shape)
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: int = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


class PromptedTransformer(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg

        num_tokens = self.prompt_config["num_tokens"]
        # num_tokens = 80
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        patch_size = _pair(patch_size)
        if self.prompt_config["initiation"] == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config["deep"]:  # noqa

                total_d_layer = layers - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_tokens):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # incorporate prompt
        B = x.shape[1]
        x = torch.cat((
            x[:1, :, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).permute(1, 0, 2),
            x[1:, :, :]
        ), dim=0)

        if self.prompt_config["deep"]:
            x = self.forward_deep_prompt(x, attn_mask)
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
        return x


class TextGuidedPromptedTransformer(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        in_dim, hidden_dim, bottleneck_dim = 512, 1024, prompt_dim
        self.vpt_generator = nn.Sequential(*[
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        ])
        self.prompt_config = vpt_cfg
        for m in self.vpt_generator:
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.prompt_config["deep"]:
            self.deep_vpt_num = layers - 1
            for i in range(layers - 1):
                setattr(self, f'vpt_generator_deep{i}', nn.Sequential(*[
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, bottleneck_dim),
                    ]))

            for i in range(layers - 1):
                for m in getattr(self, f'vpt_generator_deep{i}'):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        # initiate prompt:
        # patch_size = _pair(patch_size)
        # if self.prompt_config["initiation"] == "random":
        #     val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        #
        #     self.prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        #
        #     if self.prompt_config["deep"]:  # noqa
        #
        #         total_d_layer = layers - 1
        #         self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #             total_d_layer, num_tokens, prompt_dim))
        #         # xavier_uniform initialization
        #         nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # else:
        #     raise ValueError("Other initiation scheme is not supported")

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, text, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= num_layers-1:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        getattr(self, f'vpt_generator_deep{i-1}')(text)).expand(B, -1, -1)).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_tokens):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, text: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # incorporate prompt
        B = x.shape[1]
        # ipdb.set_trace()
        x = torch.cat((
            x[:1, :, :],
            self.prompt_dropout(self.prompt_proj(self.vpt_generator(text)).expand(B, -1, -1)).permute(1, 0, 2),
            x[1:, :, :]
        ), dim=0)

        if self.prompt_config["deep"]:
            x = self.forward_deep_prompt(x, text, attn_mask)
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
        return x


class TextGuidedPromptedTransformer2(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        in_dim, hidden_dim, bottleneck_dim = 512, 1024, prompt_dim
        self.vpt_generator = nn.Sequential(*[
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        ])
        for m in self.vpt_generator:
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # self.vpt_generator_deep = [nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]) for _ in range(layers - 1)]
        if self.prompt_config["deep"]:
            self.deep_vpt_num = layers - 1
            for i in range(layers - 1):
                setattr(self, f'vpt_generator_deep{i}', nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, bottleneck_dim),
            ]))
            for i in range(layers - 1):
                for m in getattr(self, f'vpt_generator_deep{i}'):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        # initiate prompt:
        # patch_size = _pair(patch_size)
        # if self.prompt_config["initiation"] == "random":
        #     val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        #
        #     self.prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        #
        #     if self.prompt_config["deep"]:  # noqa
        #
        #         total_d_layer = layers - 1
        #         self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #             total_d_layer, num_tokens, prompt_dim))
        #         # xavier_uniform initialization
        #         nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # else:
        #     raise ValueError("Other initiation scheme is not supported")

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, text, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= num_layers-1:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        getattr(self, f'vpt_generator_deep{i-1}')(text)).unsqueeze(1)).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_tokens):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, text: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # incorporate prompt
        B = x.shape[1]
        # ipdb.set_trace()
        x = torch.cat((
            x[:1, :, :],
            self.prompt_dropout(self.prompt_proj(self.vpt_generator(text)).unsqueeze(1)).permute(1, 0, 2),
            x[1:, :, :]
        ), dim=0)

        if self.prompt_config["deep"]:
            x = self.forward_deep_prompt(x, text, attn_mask)
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
        return x



class TextGuidedPromptedTransformer3(nn.Module):
    # generate multiple VPT vectors from a single text vector
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # 0) only transformer no linear layer 
        # 1) use two tokens to generate the vpt tokens
        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        in_dim, hidden_dim, bottleneck_dim = 512, 1024, prompt_dim

        if self.num_tokens == 1:
            in_dim = in_dim * 2


        if self.num_tokens > 0:
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            # self.vpt_generator = nn.Sequential(*[
            #     nn.Linear(in_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, bottleneck_dim*num_mapped_vpt),
            # ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -1: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -2: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -4: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -5: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)


        else: # for situation where the mlp is removed, used before rebuttal in January
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # for i in range(self.prompt_config["num_vpt"]):
        #     setattr(self, f'vpt_generator_{i}', nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]))
        # for i in range(self.prompt_config["num_vpt"]):
        #     for m in getattr(self, f'vpt_generator_{i}'):
        #         if isinstance(m, nn.Linear):
        #             torch.nn.init.trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)


        # self.vpt_generator_deep = [nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]) for _ in range(layers - 1)]
        if self.prompt_config["deep"]:
            self.deep_vpt_num = layers - 1
            for i in range(layers - 1):
                setattr(self, f'vpt_generator_deep{i}', nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ]))
            for i in range(layers - 1):
                for m in getattr(self, f'vpt_generator_deep{i}'):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        # initiate prompt:
        # patch_size = _pair(patch_size)
        # if self.prompt_config["initiation"] == "random":
        #     val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        #
        #     self.prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        #
        #     if self.prompt_config["deep"]:  # noqa
        #
        #         total_d_layer = layers - 1
        #         self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #             total_d_layer, num_tokens, prompt_dim))
        #         # xavier_uniform initialization
        #         nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # else:
        #     raise ValueError("Other initiation scheme is not supported")

        # ipdb.set_trace()

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, text, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= num_layers-1:
                    # ipdb.set_trace()
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        getattr(self, f'vpt_generator_deep{i-1}')(text).reshape(B, -1, self.width))).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_mapped_vpt):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, text: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        
        if text is not None:
            # incorporate prompt
            B = x.shape[1]
            # ipdb.set_trace()
            x = torch.cat((
                x[:1, :, :],
                self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2),
                x[1:, :, :]
            ), dim=0)

            if self.prompt_config["deep"]:
                x = self.forward_deep_prompt(x, text, attn_mask)
            else:
                for r in self.resblocks:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                        x = checkpoint(r, x, None, None, attn_mask)
                    else:
                        x = r(x, attn_mask=attn_mask)
            return x

        else: 
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x




class TextGuidedPromptedTransformer3_late(nn.Module):
    # generate multiple VPT vectors from a single text vector
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # 0) only transformer no linear layer 
        # 1) use two tokens to generate the vpt tokens
        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        in_dim, hidden_dim, bottleneck_dim = 512, 1024, prompt_dim

        if self.num_tokens == 1:
            in_dim = in_dim * 2


        if self.num_tokens > 0:
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            # self.vpt_generator = nn.Sequential(*[
            #     nn.Linear(in_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, bottleneck_dim*num_mapped_vpt),
            # ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -1: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -2: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -4: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -5: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)


        else: # for situation where the mlp is removed, used before rebuttal in January
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # for i in range(self.prompt_config["num_vpt"]):
        #     setattr(self, f'vpt_generator_{i}', nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]))
        # for i in range(self.prompt_config["num_vpt"]):
        #     for m in getattr(self, f'vpt_generator_{i}'):
        #         if isinstance(m, nn.Linear):
        #             torch.nn.init.trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)


        # self.vpt_generator_deep = [nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]) for _ in range(layers - 1)]
        if self.prompt_config["deep"]:
            self.deep_vpt_num = layers - 1
            for i in range(layers - 1):
                setattr(self, f'vpt_generator_deep{i}', nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ]))
            for i in range(layers - 1):
                for m in getattr(self, f'vpt_generator_deep{i}'):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        # initiate prompt:
        # patch_size = _pair(patch_size)
        # if self.prompt_config["initiation"] == "random":
        #     val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        #
        #     self.prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        #
        #     if self.prompt_config["deep"]:  # noqa
        #
        #         total_d_layer = layers - 1
        #         self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #             total_d_layer, num_tokens, prompt_dim))
        #         # xavier_uniform initialization
        #         nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # else:
        #     raise ValueError("Other initiation scheme is not supported")

        # ipdb.set_trace()

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, text, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= num_layers-1:
                    # ipdb.set_trace()
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        getattr(self, f'vpt_generator_deep{i-1}')(text).reshape(B, -1, self.width))).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_mapped_vpt):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, text: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        
        if text is not None:

            B = x.shape[1]
            hidden_states = None
            num_layers = self.layers

            for i in range(num_layers):
                if i == 0:
                    hidden_states = self.resblocks[i](x, attn_mask=attn_mask)
                else:
                    if i == num_layers - 1:
                        # incorporate prompt
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2)
                        hidden_states = torch.cat((
                            hidden_states[:1, :, :],
                            deep_prompt_emb,
                            hidden_states[1:, :, :]
                        ), dim=0)

                    hidden_states = self.resblocks[i](hidden_states, attn_mask)
            return hidden_states
            # # incorporate prompt
            # B = x.shape[1]
            # # ipdb.set_trace()
            # x = torch.cat((
            #     x[:1, :, :],
            #     self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2),
            #     x[1:, :, :]
            # ), dim=0)

            # if self.prompt_config["deep"]:
            #     x = self.forward_deep_prompt(x, text, attn_mask)
            # else:
            #     for r in self.resblocks:
            #         if self.grad_checkpointing and not torch.jit.is_scripting():
            #             # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            #             x = checkpoint(r, x, None, None, attn_mask)
            #         else:
            #             x = r(x, attn_mask=attn_mask)
            # return x

        else: 
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x


from open_clip.attention_pool import AttentionPool
class TextGuidedPromptedTransformer3_attn_pool(nn.Module):
    # generate multiple VPT vectors from a single text vector
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # 0) only transformer no linear layer 
        # 1) use two tokens to generate the vpt tokens
        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        in_dim, hidden_dim, bottleneck_dim = 512, 1024, prompt_dim

        if self.num_tokens == 1:
            in_dim = in_dim * 2


        if self.num_tokens > 0:
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            # self.vpt_generator = nn.Sequential(*[
            #     nn.Linear(in_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, bottleneck_dim*num_mapped_vpt),
            # ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -1: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -2: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -4: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -5: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)


        else: # for situation where the mlp is removed, used before rebuttal in January
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # for i in range(self.prompt_config["num_vpt"]):
        #     setattr(self, f'vpt_generator_{i}', nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]))
        # for i in range(self.prompt_config["num_vpt"]):
        #     for m in getattr(self, f'vpt_generator_{i}'):
        #         if isinstance(m, nn.Linear):
        #             torch.nn.init.trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)


        # self.vpt_generator_deep = [nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]) for _ in range(layers - 1)]
        if self.prompt_config["deep"]:
            self.deep_vpt_num = layers - 1
            for i in range(layers - 1):
                setattr(self, f'vpt_generator_deep{i}', nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ]))
            for i in range(layers - 1):
                for m in getattr(self, f'vpt_generator_deep{i}'):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

        # self.prompt_attn_pool = AttentionPool(
        #     768,
        #     num_heads=heads,
        #     mlp_ratio=mlp_ratio,
        #     norm_layer=norm_layer,
        #     act_layer=act_layer,
        #     latent_len=1,
        # )
        
        
        # initiate prompt:
        # patch_size = _pair(patch_size)
        # if self.prompt_config["initiation"] == "random":
        #     val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        #
        #     self.prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        #
        #     if self.prompt_config["deep"]:  # noqa
        #
        #         total_d_layer = layers - 1
        #         self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #             total_d_layer, num_tokens, prompt_dim))
        #         # xavier_uniform initialization
        #         nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # else:
        #     raise ValueError("Other initiation scheme is not supported")

        # ipdb.set_trace()

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, text, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= num_layers-1:
                    # ipdb.set_trace()
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        getattr(self, f'vpt_generator_deep{i-1}')(text).reshape(B, -1, self.width))).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_mapped_vpt):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def prompt_attn_pool_2(self, x, deep_prompt_emb):
        # x shape: [B, N, C] - query tensor
        # deep_prompt_emb shape: [B, 1, C] - key and value tensor (image patch tokens)
        
        # Compute attention scores between x and deep_prompt_emb
        # Scale dot-product attention
        scale = x.shape[-1] ** -0.5  # Scaling factor: 1/sqrt(C)
        
        # Transpose deep_prompt_emb for key
        key = deep_prompt_emb.transpose(-2, -1)  # [B, C, 1]

        # ipdb.set_trace()
        
        # Compute attention scores: [B, N, C] @ [B, C, 1] -> [B, N, 1]
        attn_scores = torch.bmm(x, key) * scale

        # ipdb.set_trace()
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]

        # ipdb.set_trace()
        
        # Apply attention weights to deep_prompt_emb (value)
        # First, transpose attn_weights to [B, 1, N]
        attn_weights = attn_weights.transpose(1, 2)  # [B, 1, N]

        # ipdb.set_trace()
        
        # Then, compute weighted sum: [B, 1, N] @ [B, N, C] -> [B, 1, C]
        out = torch.bmm(attn_weights, x)

        # ipdb.set_trace()
        
        return out

    def forward(self, x: torch.Tensor, text: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        
        if text is not None:
            for r in self.resblocks:
                x = r(x, attn_mask=attn_mask)
            B = x.shape[1] # N x B x C
            # ipdb.set_trace()
            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2)
            # 1 x B x C
            x =x.permute(1, 0, 2) # B x N x C
            # ipdb.set_trace()
            deep_prompt_emb = deep_prompt_emb.permute(1, 0, 2) # B x 1 x C
            # ipdb.set_trace()
            x = self.prompt_attn_pool_2(x, deep_prompt_emb) # B x 1 x C
            # ipdb.set_trace()
            x = x.permute(1, 0, 2) # 1 x B x C
            return x
            # # incorporate prompt
            # B = x.shape[1]
            # # ipdb.set_trace()
            # x = torch.cat((
            #     x[:1, :, :],
            #     self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2),
            #     x[1:, :, :]
            # ), dim=0)

            # if self.prompt_config["deep"]:
            #     x = self.forward_deep_prompt(x, text, attn_mask)
            # else:
            #     for r in self.resblocks:
            #         if self.grad_checkpointing and not torch.jit.is_scripting():
            #             # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            #             x = checkpoint(r, x, None, None, attn_mask)
            #         else:
            #             x = r(x, attn_mask=attn_mask)
            # return x

        else: 
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x


class TextGuidedPromptedTransformer3_attn_pool2(nn.Module):
    # generate multiple VPT vectors from a single text vector
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # 0) only transformer no linear layer 
        # 1) use two tokens to generate the vpt tokens
        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        in_dim, hidden_dim, bottleneck_dim = 512, 1024, prompt_dim

        if self.num_tokens == 1:
            in_dim = in_dim * 2


        if self.num_tokens > 0:
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            # self.vpt_generator = nn.Sequential(*[
            #     nn.Linear(in_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, bottleneck_dim*num_mapped_vpt),
            # ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -1: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -2: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -4: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -5: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)


        else: # for situation where the mlp is removed, used before rebuttal in January
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # for i in range(self.prompt_config["num_vpt"]):
        #     setattr(self, f'vpt_generator_{i}', nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]))
        # for i in range(self.prompt_config["num_vpt"]):
        #     for m in getattr(self, f'vpt_generator_{i}'):
        #         if isinstance(m, nn.Linear):
        #             torch.nn.init.trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)


        # self.vpt_generator_deep = [nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]) for _ in range(layers - 1)]
        if self.prompt_config["deep"]:
            self.deep_vpt_num = layers - 1
            for i in range(layers - 1):
                setattr(self, f'vpt_generator_deep{i}', nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ]))
            for i in range(layers - 1):
                for m in getattr(self, f'vpt_generator_deep{i}'):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

        # self.prompt_attn_pool = AttentionPool(
        #     768,
        #     num_heads=heads,
        #     mlp_ratio=mlp_ratio,
        #     norm_layer=norm_layer,
        #     act_layer=act_layer,
        #     latent_len=1,
        # )


        # Increase capacity with projection layers
        # Define projections (these would typically be defined in __init__)
        prompt_dim = 768
        self.prompt_q_proj = nn.Linear(prompt_dim, prompt_dim)
        self.prompt_k_proj = nn.Linear(prompt_dim, prompt_dim)
        self.prompt_v_proj = nn.Linear(prompt_dim, prompt_dim)
        self.prompt_out_proj = nn.Linear(prompt_dim, prompt_dim)
        
        
        # initiate prompt:
        # patch_size = _pair(patch_size)
        # if self.prompt_config["initiation"] == "random":
        #     val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        #
        #     self.prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        #
        #     if self.prompt_config["deep"]:  # noqa
        #
        #         total_d_layer = layers - 1
        #         self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #             total_d_layer, num_tokens, prompt_dim))
        #         # xavier_uniform initialization
        #         nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # else:
        #     raise ValueError("Other initiation scheme is not supported")

        # ipdb.set_trace()

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, text, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= num_layers-1:
                    # ipdb.set_trace()
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        getattr(self, f'vpt_generator_deep{i-1}')(text).reshape(B, -1, self.width))).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_mapped_vpt):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def prompt_attn_pool_2(self, x, deep_prompt_emb):
        # x shape: [B, N, C] - key and value tensor
        # deep_prompt_emb shape: [B, 1, C] - query tensor
        
        # Get dimensions
        batch_size, seq_len, dim = x.shape
        _, prompt_len, _ = deep_prompt_emb.shape
        
        # Project queries, keys, and values
        q = self.prompt_q_proj(deep_prompt_emb)  # [B, 1, C]
        k = self.prompt_k_proj(x)  # [B, N, C]
        v = self.prompt_v_proj(x)  # [B, N, C]
        
        # Multi-head attention (optional enhancement)
        # For simplicity, using 8 heads
        num_heads = 8
        head_dim = dim // num_heads
        
        # Reshape for multi-head attention
        q = q.view(batch_size, prompt_len, num_heads, head_dim).transpose(1, 2)  # [B, H, 1, D]
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, N, D]
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, N, D]
        
        # Compute scaled dot-product attention
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, 1, N]
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, N]
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # [B, H, 1, D]
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, prompt_len, dim)  # [B, 1, C]
        
        # Final projection
        out = self.prompt_out_proj(context)  # [B, 1, C]
        
        # # Repeat to match x's shape
        # out = out.repeat(1, seq_len, 1)  # [B, N, C]
        
        # # Add residual connection to x
        # out = out + x
        
        return out

    def forward(self, x: torch.Tensor, text: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        
        if text is not None:
            for r in self.resblocks:
                x = r(x, attn_mask=attn_mask)
            B = x.shape[1] # N x B x C
            # ipdb.set_trace()
            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2)
            # 1 x B x C
            x =x.permute(1, 0, 2) # B x N x C
            # ipdb.set_trace()
            deep_prompt_emb = deep_prompt_emb.permute(1, 0, 2) # B x 1 x C
            # ipdb.set_trace()
            x = self.prompt_attn_pool_2(x, deep_prompt_emb) # B x 1 x C
            # ipdb.set_trace()
            x = x.permute(1, 0, 2) # 1 x B x C
            return x
            # # incorporate prompt
            # B = x.shape[1]
            # # ipdb.set_trace()
            # x = torch.cat((
            #     x[:1, :, :],
            #     self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2),
            #     x[1:, :, :]
            # ), dim=0)

            # if self.prompt_config["deep"]:
            #     x = self.forward_deep_prompt(x, text, attn_mask)
            # else:
            #     for r in self.resblocks:
            #         if self.grad_checkpointing and not torch.jit.is_scripting():
            #             # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            #             x = checkpoint(r, x, None, None, attn_mask)
            #         else:
            #             x = r(x, attn_mask=attn_mask)
            # return x

        else: 
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x


class TextGuidedPromptedTransformer3_attn_pool3(nn.Module):
    # generate multiple VPT vectors from a single text vector
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # 0) only transformer no linear layer 
        # 1) use two tokens to generate the vpt tokens
        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        in_dim, hidden_dim, bottleneck_dim = 512, 1024, prompt_dim

        if self.num_tokens == 1:
            in_dim = in_dim * 2


        if self.num_tokens > 0:
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            # self.vpt_generator = nn.Sequential(*[
            #     nn.Linear(in_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(hidden_dim, bottleneck_dim*num_mapped_vpt),
            # ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -1: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -2: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -4: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.num_tokens == -5: 
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)


        else: # for situation where the mlp is removed, used before rebuttal in January
            num_mapped_vpt = self.prompt_config["num_mapped_vpt"]
            self.num_mapped_vpt = num_mapped_vpt
            self.vpt_generator = nn.Sequential(*[
                nn.Linear(in_dim, bottleneck_dim*num_mapped_vpt),
            ])
            for m in self.vpt_generator:
                if isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # for i in range(self.prompt_config["num_vpt"]):
        #     setattr(self, f'vpt_generator_{i}', nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]))
        # for i in range(self.prompt_config["num_vpt"]):
        #     for m in getattr(self, f'vpt_generator_{i}'):
        #         if isinstance(m, nn.Linear):
        #             torch.nn.init.trunc_normal_(m.weight, std=.02)
        #             if isinstance(m, nn.Linear) and m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)


        # self.vpt_generator_deep = [nn.Sequential(*[
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, bottleneck_dim),
        # ]) for _ in range(layers - 1)]
        if self.prompt_config["deep"]:
            self.deep_vpt_num = layers - 1
            for i in range(layers - 1):
                setattr(self, f'vpt_generator_deep{i}', nn.Sequential(*[
                nn.Linear(in_dim, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, hidden_dim*num_mapped_vpt),
                nn.GELU(),
                nn.Linear(hidden_dim*num_mapped_vpt, bottleneck_dim*num_mapped_vpt),
            ]))
            for i in range(layers - 1):
                for m in getattr(self, f'vpt_generator_deep{i}'):
                    if isinstance(m, nn.Linear):
                        torch.nn.init.trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

        # self.prompt_attn_pool = AttentionPool(
        #     768,
        #     num_heads=heads,
        #     mlp_ratio=mlp_ratio,
        #     norm_layer=norm_layer,
        #     act_layer=act_layer,
        #     latent_len=1,
        # )
        
        
        # initiate prompt:
        # patch_size = _pair(patch_size)
        # if self.prompt_config["initiation"] == "random":
        #     val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        #
        #     self.prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        #
        #     if self.prompt_config["deep"]:  # noqa
        #
        #         total_d_layer = layers - 1
        #         self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #             total_d_layer, num_tokens, prompt_dim))
        #         # xavier_uniform initialization
        #         nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # else:
        #     raise ValueError("Other initiation scheme is not supported")

        # ipdb.set_trace()

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, text, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= num_layers-1:
                    # ipdb.set_trace()
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        getattr(self, f'vpt_generator_deep{i-1}')(text).reshape(B, -1, self.width))).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_mapped_vpt):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def prompt_attn_pool_2(self, x, deep_prompt_emb):
        # x shape: [B, N, C] - query tensor
        # deep_prompt_emb shape: [B, 1, C] - key and value tensor (image patch tokens)
        
        # Compute attention scores between x and deep_prompt_emb
        # Scale dot-product attention
        scale = x.shape[-1] ** -0.5  # Scaling factor: 1/sqrt(C)
        
        # Transpose deep_prompt_emb for key
        key = deep_prompt_emb.transpose(-2, -1)  # [B, C, 1]

        # ipdb.set_trace()
        
        # Compute attention scores: [B, N, C] @ [B, C, 1] -> [B, N, 1]
        attn_scores = torch.bmm(x, key) * scale

        # ipdb.set_trace()
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]

        # ipdb.set_trace()
        
        # Apply attention weights to deep_prompt_emb (value)
        # First, transpose attn_weights to [B, 1, N]
        attn_weights = attn_weights.transpose(1, 2)  # [B, 1, N]

        # ipdb.set_trace()
        
        # Then, compute weighted sum: [B, 1, N] @ [B, N, C] -> [B, 1, C]
        out = torch.bmm(attn_weights, x)

        # ipdb.set_trace()
        
        return out

    def forward(self, x: torch.Tensor, text: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        
        if text is not None:
            for r in self.resblocks:
                x = r(x, attn_mask=attn_mask)
            B = x.shape[1] # N x B x C
            # ipdb.set_trace()
            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2)
            # 1 x B x C
            x =x.permute(1, 0, 2) # B x N x C
            # ipdb.set_trace()
            deep_prompt_emb = deep_prompt_emb.permute(1, 0, 2) # B x 1 x C
            # ipdb.set_trace()
            x = self.prompt_attn_pool_2(x, deep_prompt_emb) # B x 1 x C
            # ipdb.set_trace()
            x = x.permute(1, 0, 2) # 1 x B x C
            return x
            # # incorporate prompt
            # B = x.shape[1]
            # # ipdb.set_trace()
            # x = torch.cat((
            #     x[:1, :, :],
            #     self.prompt_dropout(self.prompt_proj(self.vpt_generator(text).reshape(B, -1, self.width))).permute(1, 0, 2),
            #     x[1:, :, :]
            # ), dim=0)

            # if self.prompt_config["deep"]:
            #     x = self.forward_deep_prompt(x, text, attn_mask)
            # else:
            #     for r in self.resblocks:
            #         if self.grad_checkpointing and not torch.jit.is_scripting():
            #             # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            #             x = checkpoint(r, x, None, None, attn_mask)
            #         else:
            #             x = r(x, attn_mask=attn_mask)
            # return x

        else: 
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x





class AddPromptedTransformer(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg

        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        patch_size = _pair(patch_size)
        if self.prompt_config["initiation"] == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config["deep"]:  # noqa

                total_d_layer = layers - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)

                    # hidden_states = torch.cat((
                    #     hidden_states[:1, :, :],
                    #     deep_prompt_emb,
                    #     hidden_states[(1+self.num_tokens):, :, :]
                    # ), dim=0)
                    hidden_states[1:,:,:] = hidden_states[1:,:,:] + deep_prompt_emb

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # incorporate prompt
        B = x.shape[1]
        # x = torch.cat((
        #     x[:1, :, :],
        #     self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).permute(1, 0, 2),
        #     x[1:, :, :]
        # ), dim=0)
        x[1:, :, :] = x[1:, :, :] + self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).permute(1, 0, 2)

        if self.prompt_config["deep"]:
            x = self.forward_deep_prompt(x, attn_mask)
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
        return x


def merge_two_tensors(x, y, dim=0, rank=1):
    assert x.shape == y.shape
    # old
    # if dim == 0:
    #     sliced_x = [x[i, :].unsqueeze(0) for i in range(x.shape[0])]
    #     sliced_y = [y[i, :].unsqueeze(0) for i in range(y.shape[0])]
    # elif dim == 1:
    #     sliced_x = [x[:, i, :].unsqueeze(1) for i in range(x.shape[1])]
    #     sliced_y = [y[:, i, :].unsqueeze(1) for i in range(y.shape[1])]
    # else:
    #     sliced_x = [x[:, :, i].unsqueeze(2) for i in range(x.shape[2])]
    #     sliced_y = [y[:, :, i].unsqueeze(2) for i in range(y.shape[2])]
    # res = []
    # for i in range(x.shape[dim]):
    #     if rank:
    #         res.append(sliced_x[i])
    #         res.append(sliced_y[i])
    #     else:
    #         res.append(sliced_y[i])
    #         res.append(sliced_x[i])
    # new
    x_dim = list(x.shape)
    x_dim[dim] = x_dim[dim]*2
    z = torch.zeros(x_dim, dtype=x.dtype, device=x.device)
    res = [y, x]
    if dim == 0:
        z[::2, :] = res[rank]
        z[1::2, :] = res[1 - rank]
    elif dim == 1:
        z[:, ::2, :] = res[rank]
        z[:, 1::2, :] = res[1 - rank]
    else:
        z[:, :, ::2] = res[rank]
        z[:, :, 1::2] = res[1 - rank]
    return z


def get_odd_channels(x, dim=0, odd=0):
    # res = []
    # for i in range(x.shape[dim]):
    #     if i % 2 == odd:
    #         if dim == 0:
    #             res.append(x[i, :, :].unsqueeze(0))
    #         elif dim == 1:
    #             res.append(x[:, i, :].unsqueeze(1))
    #         else:
    #             res.append(x[:, :, i].unsqueeze(2))
    # return torch.cat(res, dim=dim)
    if dim == 0:
        return x[odd::2, :]
    elif dim == 1:
        return x[:, odd::2, :]
    else:
        return x[:, :, odd::2]


class PositionalPromptedTransformer(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg

        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])
        self.vpt_position = self.prompt_config["vpt_position"]

        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        patch_size = _pair(patch_size)
        if self.prompt_config["initiation"] == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            # self.prompt_embeddings = nn.Parameter(torch.zeros(
            #     1, num_tokens, prompt_dim))
            # # xavier_uniform initialization
            # nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config["deep"]:  # noqa

                total_d_layer = layers - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype


    def forward_deep_prompt(self, embedding_output, attn_mask):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                    if self.vpt_position == 1:
                        hidden_states = torch.cat((
                            hidden_states[:1, :, :],
                            deep_prompt_emb,
                            hidden_states[(1+self.num_tokens):, :, :]
                        ), dim=0)
                    elif self.vpt_position == 2:
                        hidden_states = torch.cat((
                            hidden_states[:-self.num_tokens, :, :],
                            deep_prompt_emb
                        ), dim=0)
                    elif self.vpt_position == 3 or self.vpt_position == 4:
                        hidden_states = torch.cat((
                            hidden_states[:1, :, :],
                            merge_two_tensors(get_odd_channels(hidden_states[1:, :, :], odd=0, dim=0), deep_prompt_emb, dim=0)
                        ), dim=0)
                    else:
                        raise ValueError("Unsupported VPT position")

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # incorporate prompt
        # B = x.shape[1]
        # x = torch.cat((
        #     x[:1, :, :],
        #     self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).permute(1, 0, 2),
        #     x[1:, :, :]
        # ), dim=0)

        if self.prompt_config["deep"]:
            x = self.forward_deep_prompt(x, attn_mask)
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                    x = checkpoint(r, x, None, None, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
        return x


class LatentPromptedTransformer(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Latent Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + width))  # noqa
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, 196, width))
        # xavier_uniform initialization
        # nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # incorporate prompt
        B = x.shape[1]
        x[1:,:,:] = x[1:,:,:] + self.prompt_embeddings.expand(B, -1, -1).permute(1, 0, 2)

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class PerCatePromptedTransformer(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            vpt_cfg: dict = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg

        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])

        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        patch_size = _pair(patch_size)
        if self.prompt_config["initiation"] == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                vpt_cfg["num_cate"], num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config["deep"]:  # noqa

                total_d_layer = layers - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, vpt_cfg["num_cate"], num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

        self.cate_num = vpt_cfg["num_cate"]


    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward_deep_prompt(self, embedding_output, cate_label=None, attn_mask=None):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[1]
        num_layers = self.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.resblocks[i](embedding_output, attn_mask=attn_mask)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1][cate_label])).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1+self.num_tokens):, :, :]
                    ), dim=0)

                hidden_states = self.resblocks[i](hidden_states, attn_mask)
        return hidden_states

    def forward(self, x: torch.Tensor, cate_label: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if cate_label is not None:
        # incorporate prompt
            x = torch.cat((
                x[:1, :, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[cate_label,:,:])).permute(1, 0, 2),
                x[1:, :, :]
            ), dim=0)

            if self.prompt_config["deep"]:
                x = self.forward_deep_prompt(x, cate_label, attn_mask)
            else:
                for r in self.resblocks:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                        x = checkpoint(r, x, None, None, attn_mask)
                    else:
                        x = r(x, attn_mask=attn_mask)
            return x
        else:
            B = x.shape[1]
            all_res = []
            for i in range(self.cate_num):
                cate_label = torch.ones(B).type(torch.long).cuda()*i

                x1 = torch.cat((
                    x[:1, :, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[cate_label, :, :])).permute(1, 0, 2),
                    x[1:, :, :]
                ), dim=0)

                if self.prompt_config["deep"]:
                    x1 = self.forward_deep_prompt(x1, cate_label, attn_mask)
                else:
                    for r in self.resblocks:
                        x1 = r(x1, attn_mask=attn_mask)
                all_res.append(x1)

            res = torch.cat(all_res, dim=1)
            return res


class PromptedVisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            vpt_cfg: dict = None
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        if vpt_cfg["vpt_type"] == "shared":
            self.transformer = PromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "percate":
            self.transformer = PerCatePromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "latent":
            self.transformer = LatentPromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "add":
            self.transformer = LatentPromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        # elif vpt_cfg["vpt_type"] in ["text_guided", "text_guided_late", "text_guided_attn_pool"]:
        elif vpt_cfg["vpt_type"] in ["text_guided"]:
            self.transformer = TextGuidedPromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        else:
            raise ValueError("Unsupported vpt type!")

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if isinstance(self.transformer, PerCatePromptedTransformer):
            x = self.transformer(x, label)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
            # print('pooled', pooled.shape, 'tokens', tokens.shape)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled


class TextGuidedPromptedVisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            vpt_cfg: dict = None
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        
        if vpt_cfg["vpt_type"] == "text_guided":
            self.transformer = TextGuidedPromptedTransformer3(  
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )

        elif vpt_cfg["vpt_type"] == "text_guided_late":
            self.transformer = TextGuidedPromptedTransformer3_late(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "text_guided_attn_pool":
            self.transformer = TextGuidedPromptedTransformer3_attn_pool(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "text_guided_attn_pool2":
            self.transformer = TextGuidedPromptedTransformer3_attn_pool2(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "text_guided_attn_pool3":
            self.transformer = TextGuidedPromptedTransformer3_attn_pool3(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor, text: torch.Tensor, label: Optional[torch.Tensor] = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, text)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
            # print('pooled', pooled.shape, 'tokens', tokens.shape)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled


class OWPromptedVisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            vpt_cfg: dict = None
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        self.vpt_cfg = vpt_cfg
        if vpt_cfg["vpt_type"] == "shared" or vpt_cfg["vpt_type"] == "ow":
            self.transformer = PromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "ow_text":
            self.transformer = TextGuidedPromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "percate":
            self.transformer = PerCatePromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "latent":
            self.transformer = LatentPromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        else:
            raise ValueError("Unsupported vpt type!")

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        # if vpt_cfg["vpt_mapper"] == "":
        #     self.mapper =

        # load known and novel embeddings
        self.known_embed = torch.from_numpy(np.load('/home/ypliu/projects/OccludedCLIP/open_clip-main/embeddings/known_txt_embedd.npy').transpose()).cuda()
        self.novel_embed = torch.from_numpy(np.load('/home/ypliu/projects/OccludedCLIP/open_clip-main/embeddings/novel_txt_embedd.npy').transpose()).cuda()
        self.known_embed.detach()
        self.novel_embed.detach()
        self.init_parameters()
        
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if isinstance(self.transformer, PerCatePromptedTransformer):
            x = self.transformer(x, label)
        elif isinstance(self.transformer, TextGuidedPromptedTransformer):
            # check known or novel
            known_flag = open('embeddings/known_flag.txt').readlines()[0].strip()
            if known_flag == 'known':
                text_feat = self.known_embed
            elif known_flag == 'novel':
                text_feat = self.novel_embed
            else:
                raise ValueError('Wrong embedding type!')
            x = self.transformer(x, text_feat)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            # pooled, tokens = self._global_pool(x)
            _, pooled = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        # print('pooled.shape')
        # print(pooled.shape)
        return pooled[:,:self.vpt_cfg['num_tokens']+1]


class PositionalPromptedVisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            vpt_cfg: dict = None
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # ------------------------- Visual Prompt Tuning -------------------------
        self.prompt_config = vpt_cfg
        num_tokens = self.prompt_config["num_tokens"]
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(self.prompt_config['dropout'])
        self.vpt_position = self.prompt_config["vpt_position"]
        pos_embed_type = self.prompt_config["pos_embed_type"]
        self.pos_embed_type = pos_embed_type
        # ------------------------- Visual Prompt Tuning -------------------------

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
            if self.vpt_position in [1,2,3]:
                self.prompt_positional_embedding = nn.Parameter(
                    scale * torch.randn(num_tokens, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1 + num_tokens, width), requires_grad=False)
            print('self.grid_size[0]', self.grid_size[0])
            grid_size = int(math.sqrt(self.grid_size[0] * self.grid_size[1] + num_tokens))
            pos_embed_type = get_2d_sincos_pos_embed(width, grid_size, cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # ------------------------- Visual Prompt Tuning -------------------------
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        patch_size = _pair(patch_size)
        if self.prompt_config["initiation"] == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")
        # self.prompt_positional_embedding = nn.Parameter(scale * torch.randn(1, num_tokens, width))
        # ------------------------- Visual Prompt Tuning -------------------------

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        self.transformer = PositionalPromptedTransformer(
            patch_size,
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            vpt_cfg=vpt_cfg
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]

        # concat prompt embedding and prompt position embedding
        if self.vpt_position == 1:
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(x.shape[0], -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        elif self.vpt_position == 2:
            x = torch.cat((
                x,
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(x.shape[0], -1, -1)),
            ), dim=1)
        elif self.vpt_position == 3 or self.vpt_position == 4:
            x = torch.cat((
                x[:, :1, :],
                merge_two_tensors(x[:, 1:, :], self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(x.shape[0], -1, -1)),dim=1),
            ), dim=1)
        else:
            raise ValueError("Unsupported VPT position!")

        positional_embedding = self.positional_embedding.to(x.dtype)
        if self.pos_embed_type == 'learnable':
            if self.vpt_position == 1:
                positional_embedding = torch.cat((
                                            positional_embedding[:1,:],
                                            self.prompt_positional_embedding,
                                            positional_embedding[1:,:]
                                        ), dim=0)
            elif self.vpt_position == 2:
                positional_embedding = torch.cat((
                    positional_embedding,
                    self.prompt_positional_embedding
                ), dim=0)
            elif self.vpt_position == 3:
                positional_embedding = torch.cat((
                    positional_embedding[:1,:],
                    merge_two_tensors(positional_embedding[1:,:], self.prompt_positional_embedding, dim=0)
                ), dim=0)
            elif self.vpt_position == 4:
                positional_embedding = torch.cat((
                    positional_embedding[:1, :],
                    merge_two_tensors(positional_embedding[1:, :], positional_embedding[1:, :], dim=0)
                ), dim=0)
            else:
                raise ValueError("Unsupported VPT position!")
        x = x + positional_embedding

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled


class PatchPrompter(nn.Module):
    def __init__(self, cfg):
        super(PatchPrompter, self).__init__()
        self.patch_size = cfg["patch_size"]
        self.prompt_size = cfg["prompt_size"]
        self.fg_size = cfg["patch_size"] - cfg["prompt_size"] * 2

        self.prompt_patch = nn.Parameter(torch.randn([1, 3, cfg["image_size"], cfg["image_size"]]))

    def forward(self, x):
        _, _, h, w = x.size()

        fg_in_patch = torch.zeros([1, 3, self.fg_size, self.fg_size]).cuda()
        fg_in_patch = F.pad(fg_in_patch, (self.prompt_size, self.prompt_size, self.prompt_size, self.prompt_size),
                            "constant", 1)
        mask = fg_in_patch.repeat(1, 1, h // self.patch_size, w // self.patch_size)
        self.prompt = self.prompt_patch * mask

        return x + self.prompt


class DilatedPrompter(nn.Module):
    def __init__(self, image_size, cfg):
        super(DilatedPrompter, self).__init__()
        self.patch_size = cfg["patch_size"]
        self.dilate = cfg["dilate"]
        # self.prompt_size = cfg["prompt_size"]
        # self.fg_size = cfg["patch_size"] - cfg["prompt_size"] * 2

        self.prompt_patch = nn.Parameter(torch.randn([1, 3, cfg["image_size"], cfg["image_size"]]))

        fg_in_patch = torch.zeros([1, 3, self.patch_size, self.patch_size]).cuda()
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                if ((i+1) % self.dilate == 0) or ((j+1) % self.dilate == 0):
                    fg_in_patch[0, 0, i, j] = 1
                    fg_in_patch[0, 1, i, j] = 1
                    fg_in_patch[0, 2, i, j] = 1

        self.mask = fg_in_patch.repeat(1, 1, image_size // self.patch_size, image_size // self.patch_size)


    def forward(self, x):
        self.prompt = self.prompt_patch * self.mask

        return x + self.prompt


class MaskPrompter(nn.Module):
    def __init__(self, cfg):
        super(MaskPrompter, self).__init__()
        self.prompt_patch = nn.Parameter(torch.randn([1, 3, cfg["image_size"], cfg["image_size"]]))

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,3,1,1)
            prompt = self.prompt_patch * mask
            return x + prompt
        else:
            return x + self.prompt_patch


class TriMaskPrompter(nn.Module):
    def __init__(self, cfg):
        super(TriMaskPrompter, self).__init__()
        self.prompt_patch1 = nn.Parameter(torch.randn([1, 3, cfg["image_size"], cfg["image_size"]]))
        self.prompt_patch2 = nn.Parameter(torch.randn([1, 3, cfg["image_size"], cfg["image_size"]]))
        self.prompt_patch3 = nn.Parameter(torch.randn([1, 3, cfg["image_size"], cfg["image_size"]]))

    def forward(self, x, mask1=None, mask2=None, mask3=None):
        if mask1 is not None:
            mask1 = mask1.unsqueeze(1).repeat(1,3,1,1)
            prompt1 = self.prompt_patch1 * mask1
            mask2 = mask2.unsqueeze(1).repeat(1, 3, 1, 1)
            prompt2 = self.prompt_patch2 * mask2
            mask3 = mask3.unsqueeze(1).repeat(1, 3, 1, 1)
            prompt3 = self.prompt_patch3 * mask3
            return x + prompt1 + prompt2 + prompt3
        else:
            return x + self.prompt_patch1 + self.prompt_patch2 + self.prompt_patch3


class DeformPrompter(nn.Module):
    def __init__(self, image_size, cfg):
        super(DeformPrompter, self).__init__()
        self.image_size = image_size
        self.patch_size = cfg["patch_size"]
        self.prompt_size = cfg["prompt_size"]

        kernel_size = cfg["patch_size"]
        self.p_conv = nn.Conv2d(3, 2 * self.prompt_size
                                , kernel_size=kernel_size, padding=1, stride=kernel_size)

        self.prompt_patch = nn.Parameter(torch.randn([1, 3, cfg["image_size"], cfg["image_size"]]))


    def forward(self, x):
        n, _, h, w = x.size()
        prompt_offset = self.p_conv(x)
        prompt_offset = torch.clamp(prompt_offset.floor(), 0, self.patch_size-1)
        # prompt_offset = prompt_offset.reshape(n, 2, self.prompt_size)

        mask = torch.zeros([n, 3, self.image_size, self.image_size]).cuda()
        for i in range(n):
            for j in range(h // self.patch_size):
                for k in range(w // self.patch_size):
                    x_axis = (j*self.patch_size + prompt_offset[i, 0:self.prompt_size, j, k]).long()
                    y_axis = (k*self.patch_size + prompt_offset[i, self.prompt_size:, j, k]).long()
                    mask[i, :, x_axis, y_axis] = 1 - mask[i, :, x_axis, y_axis]
        self.prompt = self.prompt_patch * mask
        return x + self.prompt


class PadPrompter(nn.Module):
    def __init__(self, cfg):
        super(PadPrompter, self).__init__()
        pad_size = cfg["prompt_size"]
        image_size = cfg["image_size"]

        self.base_size = image_size - pad_size * 2
        self.prompt_pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.prompt_pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.prompt_pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))
        self.prompt_pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.prompt_pad_left, base, self.prompt_pad_right], dim=3)
        prompt = torch.cat([self.prompt_pad_up, prompt, self.prompt_pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt

from torchvision.transforms import Normalize
class EVPPrompter(nn.Module):
    def __init__(self, cfg):
        super(EVPPrompter, self).__init__()
        # image size -> 164
        pad_length = int((224 - 164) / 2)
        self.pad_dim = (pad_length, pad_length, pad_length, pad_length)
        self.perturbation = torch.nn.Parameter(torch.zeros((3, 224, 224)).float(), requires_grad=True)
        self.normalization = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    def forward(self, x):
        images = F.pad(x, self.pad_dim, "constant", value=0)
        noise = self.perturbation.repeat(images.size(0), 1, 1, 1)
        # if self.train():
        #     noise.retain_grad()

        images = self.normalization(images + noise)
        # if self.train():
        #     images.require_grad = True

        return images


class SpatialPromptedVisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            spt_cfg: dict = None
    ):
        if spt_cfg["spt_type"] == 'evp':
            image_size = 224
        
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        # different types of prompt
        spt_cfg["image_size"] = image_size
        spt_cfg["patch_size"] = patch_size
        if spt_cfg["spt_type"] == "patch":
            spt_cfg["prompt_size"] = 1
            self.prompter = PatchPrompter(spt_cfg)
        elif spt_cfg["spt_type"] == "dilate":
            self.prompter = DilatedPrompter(image_size, spt_cfg)
        elif spt_cfg["spt_type"] == "deform":
            self.prompter = DeformPrompter(image_size, spt_cfg)
        elif spt_cfg["spt_type"] == 'all':
            spt_cfg["prompt_size"] = 30
            prompter1 = PadPrompter(spt_cfg)
            spt_cfg["prompt_size"] = 1
            prompter2 = PatchPrompter(spt_cfg)
            self.prompter = nn.Sequential(prompter1, prompter2)
        elif spt_cfg["spt_type"] == 'pad':
            spt_cfg["prompt_size"] = 30
            self.prompter = PadPrompter(spt_cfg)
        elif spt_cfg["spt_type"] == 'evp':
            self.prompter = EVPPrompter(spt_cfg)
        elif spt_cfg["spt_type"] == 'mask':
            self.prompter = MaskPrompter(spt_cfg)
        elif spt_cfg["spt_type"] == 'trimask':
            self.prompter = TriMaskPrompter(spt_cfg)
        else:
            raise ValueError('Unsupported spatial prompt type!')

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x):
        if isinstance(x, list):
            if isinstance(self.prompter, MaskPrompter):
                img, obj_mask, _, _ = x
                x = self.prompter(img, obj_mask)
            elif isinstance(self.prompter, TriMaskPrompter):
                img, obj_mask, ocder_mask, ocdee_mask = x
                x = self.prompter(img, obj_mask, ocder_mask, ocdee_mask)
            else:
                raise ValueError("Input Error")
        else:
            x = self.prompter(x)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled


class SpatialVisualPromptedVisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            vpt_cfg: dict = None
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        if vpt_cfg["vpt_type"] == "shared":
            self.transformer = PromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        elif vpt_cfg["vpt_type"] == "percate":
            self.transformer = PerCatePromptedTransformer(
                patch_size,
                width,
                layers,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                vpt_cfg=vpt_cfg
            )
        else:
            raise ValueError("Unsupported vpt type!")

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        # different types of prompt
        vpt_cfg["image_size"] = image_size
        vpt_cfg["patch_size"] = patch_size
        if vpt_cfg["spt_type"] == "patch":
            vpt_cfg["prompt_size"] = 1
            self.prompter = PatchPrompter(vpt_cfg)
        elif vpt_cfg["spt_type"] == 'all':
            vpt_cfg["prompt_size"] = 30
            prompter1 = PadPrompter(vpt_cfg)
            vpt_cfg["prompt_size"] = 1
            prompter2 = PatchPrompter(vpt_cfg)
            self.prompter = nn.Sequential(prompter1, prompter2)
        else:
            raise ValueError('Unsupported spatial prompt type!')

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        x = self.prompter(x)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if isinstance(self.transformer, PerCatePromptedTransformer):
            x = self.transformer(x, label)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled