"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

from typing import Callable, Any

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

import pdb


logger = logging.getLogger(__name__)


class Functional2Layer(torch.nn.Module):
    # https://github.com/davrot/pytorch_original/blob/b354fdee89cf1df5908df614456ba422d19ae4c0/torch/nn/utils/Functional2Layer.py
    def __init__(
        self, func: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input, *self.args, **self.kwargs)

    def extra_repr(self) -> str:
        func_name = (
            self.func.__name__ if hasattr(self.func, "__name__") else str(self.func)
        )
        args_repr = ", ".join(map(repr, self.args))
        kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"func={func_name}, args=({args_repr}), kwargs={{{kwargs_repr}}}"


class CausalConv1d(torch.nn.Conv1d):
    # ref: https://github.com/pytorch/pytorch/issues/1333#issuecomment-453702879
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self._padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self._padding, 0)))


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True, skip=True, resid_pdrop=0.1):
        super(CausalConvBlock, self).__init__()
        self.skip = skip

        self.ln = nn.LayerNorm(in_ch)
        self.trans = Functional2Layer(torch.transpose, 1, 2)
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x, y=None):
        if y is None:
            y = x.clone()

        y = self.trans(self.ln(y))
        y = self.trans(self.conv(y))

        if self.skip:
            x = x + self.dropout(y)
        else:
            x = self.dropout(y)

        return x


class MLPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True, skip=True, resid_pdrop=0.1):
        super(MLPBlock, self).__init__()
        self.skip = skip

        self.ln = nn.LayerNorm(in_ch)
        self.mlp = nn.Linear(in_ch, out_ch, bias=bias)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x, y=None):
        if y is None:
            y = x.clone()

        y = self.mlp(self.ln(y))

        if self.skip:
            x = x + self.dropout(y)
        else:
            x = self.dropout(y)

        return x


def causal_moving_average(x, dim, window_size=3):
    # Permute dimensions to bring the specified dimension to the last axis for convolution
    x = x.transpose(dim, -1)

    # Create an averaging kernel of the specified window size
    kernel = torch.ones(x.shape[1], 1, window_size, device=x.device) / window_size

    # Apply 1D convolution with left padding (causal)
    # Padding only on the left to maintain causality
    padding = window_size - 1  # Left padding only
    cma = F.conv1d(x, kernel, padding=padding, groups=x.shape[1])

    # Remove extra elements caused by padding on the right
    cma = cma[..., :x.shape[-1]]

    # Restore original dimension order
    cma = cma.transpose(-1, dim)

    return cma


class WaveletConfig:

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    discrete_input = False
    input_size = 10
    n_embd = 768
    n_layer = 12

    kernel_height = 1
    kernel_width = 3

    converter = 'id'
    scaler = 'id'

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x_q, x_k=None, x_v=None):
        # x: (b, w, c)
        (
            B,
            T,
            C,
        ) = x_q.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        if x_k is None and x_v is None:
            x_k = x_q
            x_v = x_q
        else:
            assert x_k is not None and x_v is not None

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x_k).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x_q).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x_v).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))   # same as input x

        return y


class AttentionBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config, attn_type='self'):
        super().__init__()
        self.attn_type = attn_type

        if attn_type == 'self':
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.attn = CausalAttention(config)
        elif attn_type == 'cross':
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)
            self.ln3 = nn.LayerNorm(config.n_embd)
            self.attn = CausalAttention(config)
        else:
            raise ValueError(f'Unknown attention type {attn_type}.')

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, q, k_v=None):
        if k_v is None:
            assert self.attn_type == 'self'
            q = q + self.attn(self.ln1(q))
            q = q + self.mlp(self.ln2(q))
        else:
            assert self.attn_type == 'cross'
            normed_k_v = self.ln2(k_v)
            q = q + self.attn(self.ln1(q), normed_k_v, normed_k_v)
            q = q + self.mlp(self.ln3(q))

        return q


class PNet(nn.Module):
    def __init__(self, in_ch, out_ch, config):
        super().__init__()
        dilation = 2
        n_layers = math.ceil(float(config.block_size - 1) / float((config.kernel_width - 1) * dilation))

        self.net = nn.Sequential(*[CausalConvBlock(in_ch, out_ch, kernel_size=config.kernel_width, dilation=dilation,
                                                   skip=False, resid_pdrop=config.resid_pdrop) for _ in range(n_layers)])

    def forward(self, x):
        x = self.net(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, config):
        super().__init__()
        dilation = 2
        n_layers = math.ceil(float(config.block_size - 1) / float((config.kernel_width - 1) * dilation))

        self.net = nn.Sequential(*[CausalConvBlock(in_ch, out_ch, kernel_size=config.kernel_width, dilation=dilation,
                                                   skip=False, resid_pdrop=config.resid_pdrop) for _ in range(n_layers)])
    def forward(self, x):
        x = self.net(x)

        return x


class WaveletBlock(nn.Module):
    """an unassuming Wavelet block"""

    def __init__(self, in_ch, out_ch, config, block_type='analysis', scaler='id'):
        super().__init__()
        self.pnet = PNet(in_ch, out_ch, config)
        self.unet = UNet(in_ch, out_ch, config)

        self.block_type = block_type
        self.scaler = scaler

        if self.block_type == 'analysis':
            # https://medium.com/@akp83540/adaptive-average-pooling-layer-cb438d029022

            if self.scaler == 'id':
                self.a_fuse_op = nn.Identity()
            elif self.scaler == 'mlp':
                self.a_fuse_op = MLPBlock(in_ch, out_ch, resid_pdrop=config.resid_pdrop)
            elif self.scaler == 'conv':
                self.a_fuse_op = CausalConvBlock(in_ch, out_ch, config.kernel_width, resid_pdrop=config.resid_pdrop)
            elif self.scaler == 'attn':
                self.a_fuse_op = AttentionBlock(config, attn_type='self')
            else:
                raise ValueError(f'{config.scaler} is not supported.')

        if self.block_type == 'synthesis':
            self.s_fuse_op = AttentionBlock(config, attn_type='cross')

    def lifting_scheme(self, x):
        # bz, T, C = x.size()

        # # Step 1: Split the signal into even and odd indexed parts
        # even = x[:, ::2]  # even indexed elements
        # odd = x[:, 1::2]  # odd indexed elements

        even = self.a_fuse_op(x)
        odd = even.clone()

        # # Shift even tensor to the right by one position along the time dimension
        # even_shifted = torch.cat([torch.zeros_like(even[:, :1, :]), even[:, :-1, :]], dim=1)
        # odd_shifted = torch.cat([torch.zeros_like(odd[:, :1, :]), odd[:, :-1, :]], dim=1)

        # Causal Predict Step - Calculate details (high-pass filter)
        d = odd - self.pnet(even)   # Only use past values for causality
        # d = odd - self.pnet(even_shifted)   # Only use past values for causality

        # d_shifted = torch.cat([torch.zeros_like(d[:, :1, :]), d[:, :-1, :]], dim=1)

        # Causal Update Step - Calculate smooth (low-pass filter)
        s = even + self.unet(d) # Only use past values for causality
        # s = even + self.unet(d_shifted) # Only use past values for causality

        return s, d

    # Define the inverse lifting scheme process for one scale (reconstruction)
    def inverse_lifting_scheme(self, s, d):
        # Inverse Update Step - Reconstruct even part from smooth (s) and detail (d)
        # d_shifted = torch.cat([torch.zeros_like(d[:, :1, :]), d[:, :-1, :]], dim=1)

        even = s - self.unet(d)
        # even = s - self.unet(d_shifted)

        # even_shifted = torch.cat([torch.zeros_like(even[:, :1, :]), even[:, :-1, :]], dim=1)

        # Inverse Predict Step - Reconstruct odd part from detail (d) and even part
        odd = d + self.pnet(even)
        # odd = d + self.pnet(even_shifted)

        # Step 3: Merge the even and odd parts
        # B, T, C = even.shape
        # x = torch.empty(B, T * 2, C, dtype=even.dtype, device=even.device)
        # indices = torch.arange(T * 2, device=even.device).reshape(1, T * 2, 1).expand(B, T * 2, C)
        # even_indices = indices[:, ::2]  # Target positions for even tensor
        # odd_indices = indices[:, 1::2]  # Target positions for odd tensor
        # x = x.scatter(1, even_indices, even)
        # x = x.scatter(1, odd_indices, odd)

        # x = self.s_fuse_op(odd, even) # odd + attn(q=odd, v=even)
        x = self.s_fuse_op(even, odd)   # even + attn(q=even, v=odd)

        return x

    def forward(self, x):
        if self.block_type == 'analysis':
            s, d = self.lifting_scheme(x)

            return s, d
        elif self.block_type == 'synthesis':
            s, d = x
            x = self.inverse_lifting_scheme(s, d)

            return x
        else:
            raise ValueError(f'Unknown block type: {self.block_type}')

class WAVELET(nn.Module):
    """the full WAVELET model, with a context size of block_size"""

    def __init__(self, config: WaveletConfig):
        super().__init__()

        # input embedding stem
        if config.discrete_input:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        else:
            self.tok_emb = nn.Linear(config.input_size, config.n_embd)
        self.discrete_input = config.discrete_input
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # todo
        if config.converter == 'id':
            self.converter_blocks = nn.Sequential(*[nn.Identity() for _ in range(config.n_layer + 1)])
        elif config.converter == 'mlp':
            self.converter_blocks = nn.Sequential(*[MLPBlock(config.n_embd, config.n_embd,
                                                             resid_pdrop=config.resid_pdrop)
                                                    for _ in range(config.n_layer + 1)])
        elif config.converter == 'conv':
            self.converter_blocks = nn.Sequential(*[CausalConvBlock(config.n_embd, config.n_embd,
                                                                    kernel_size=config.kernel_width, stride=1,
                                                                    resid_pdrop=config.resid_pdrop)
                                                    for _ in range(config.n_layer + 1)])
        elif config.converter == 'attn':
            self.converter_blocks = nn.Sequential(*[AttentionBlock(config)
                                                    for _ in range(config.n_layer + 1)])
        else:
            raise ValueError(f'Unknown converter: {config.converter}')

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # wavelet
        self.wavelet_analysis_blocks = nn.Sequential(
            *[WaveletBlock(config.n_embd, config.n_embd, config, 'analysis', scaler=config.scaler) for _ in range(config.n_layer)])
        self.wavelet_synthesis_blocks = nn.Sequential(
            *[WaveletBlock(config.n_embd, config.n_embd, config, 'synthesis', scaler=config.scaler) for _ in range(config.n_layer)])

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, WAVELET):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, CausalConv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.ConvTranspose1d, CausalConv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root WAVELET module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )

        return optimizer

    def multi_scale_lifting(self, x, scales):
        smooth = x
        details = []

        for scale in range(scales):
            s, d = self.wavelet_analysis_blocks[scale](smooth)
            details.append(d)
            smooth = s  # Continue processing the smooth part

        return smooth, details

    def multi_scale_inverse_lifting(self, smooth, details):
        scales = len(details)

        smoothes = [smooth]
        # Start reconstruction from the final smooth coefficients
        for scale in reversed(range(scales)):
            d = details[scale]
            smooth = self.wavelet_synthesis_blocks[scale]((smooth, d))
            smoothes.append(smooth)

        return smooth, smoothes

    def forward(self, idx, targets=None):
        if self.discrete_input:
            b, t = idx.size()
        else:
            b, t, dim = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the WAVELET model
        token_embeddings = self.tok_emb(idx)    # (b, w, c), each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # (1, w, c), each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        # x = self.blocks(x)
        # x = self.ln_f(x)
        # logits = self.head(x)

        scales = len(self.wavelet_analysis_blocks)

        smooth, details = self.multi_scale_lifting(x, scales)

        # Converter
        smooth = self.converter_blocks[-1](smooth)

        # todo
        for i in range(scales):
            detail = self.converter_blocks[i](details[i])
            details[i] = detail

        merged, smoothes = self.multi_scale_inverse_lifting(smooth, details)

        merged = self.ln_f(merged)
        logits = self.head(merged)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        d_loss = torch.mean(torch.stack([F.smooth_l1_loss(detail, torch.zeros_like(detail)) for detail in details]))
        # s_loss = torch.mean(torch.stack([F.smooth_l1_loss(s_prev, simple_moving_average(s_next, dim=1)[:, ::2])
        #                                  for s_prev, s_next in zip(smoothes[:-1], smoothes[1:])]))

        s_loss = torch.mean(torch.stack([F.smooth_l1_loss(s_prev, causal_moving_average(s_next, dim=1))
                                         for s_prev, s_next in zip(smoothes[:-1], smoothes[1:])]))

        logits = logits[:, :t, :]

        return logits.contiguous(), loss, d_loss, s_loss
