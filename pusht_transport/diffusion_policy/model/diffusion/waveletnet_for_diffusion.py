from typing import Union, Optional, Tuple
import logging
from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

import math

import pdb


logger = logging.getLogger(__name__)


class Functional2Layer(nn.Module):
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


class CausalConv1d(nn.Conv1d):
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


class PNet(nn.Module):
    def __init__(self, in_ch, out_ch, horizon, kernel_width, resid_pdrop):
        super().__init__()
        dilation = 2
        n_layers = math.ceil(float(horizon - 1) / float((kernel_width - 1) * dilation))

        self.net = nn.Sequential(*[CausalConvBlock(in_ch, out_ch, kernel_size=kernel_width, dilation=dilation,
                                                   skip=False, resid_pdrop=resid_pdrop) for _ in range(n_layers)])

    def forward(self, x):
        x = self.net(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, horizon, kernel_width, resid_pdrop):
        super().__init__()
        dilation = 2
        n_layers = math.ceil(float(horizon - 1) / float((kernel_width - 1) * dilation))

        self.net = nn.Sequential(*[CausalConvBlock(in_ch, out_ch, kernel_size=kernel_width, dilation=dilation,
                                                   skip=False, resid_pdrop=resid_pdrop) for _ in range(n_layers)])

    def forward(self, x):
        x = self.net(x)

        return x


class WaveletBlock(nn.Module):
    """an unassuming Wavelet block"""

    def __init__(self, n_emb, n_head, horizon, kernel_width, p_drop_attn, block_type='analysis', scaler='id'):
        super().__init__()
        self.pnet = PNet(n_emb, n_emb, horizon, kernel_width, resid_pdrop=p_drop_attn)
        self.unet = UNet(n_emb, n_emb, horizon, kernel_width, resid_pdrop=p_drop_attn)

        self.block_type = block_type
        self.scaler = scaler

        if self.block_type == 'analysis':
            # https://medium.com/@akp83540/adaptive-average-pooling-layer-cb438d029022

            if self.scaler == 'id':
                self.a_fuse_op = nn.Identity()
            elif self.scaler == 'mlp':
                # self.a_fuse_op = MLPBlock(in_ch, out_ch, resid_pdrop=config.resid_pdrop)
                raise NotImplementedError
            elif self.scaler == 'conv':
                # self.a_fuse_op = CausalConvBlock(in_ch, out_ch, config.kernel_width, resid_pdrop=config.resid_pdrop)
                raise NotImplementedError
            elif self.scaler == 'attn':
                self.a_fuse_op = nn.TransformerDecoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True # important for stability
            )
            else:
                raise ValueError(f'{scaler} is not supported.')

        if self.block_type == 'synthesis':
            self.s_fuse_op = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )

    def lifting_scheme(self, x, memory, mask, memory_mask):
        if not isinstance(self.a_fuse_op, nn.Identity):
            even = self.a_fuse_op(
                tgt=x,
                memory=memory,
                tgt_mask=mask,
                memory_mask=memory_mask)
        else:
            even = self.a_fuse_op(x)
        odd = even.clone()

        # Causal Predict Step - Calculate details (high-pass filter)
        d = odd - self.pnet(even)   # Only use past values for causality

        # Causal Update Step - Calculate smooth (low-pass filter)
        s = even + self.unet(d) # Only use past values for causality

        return s, d

    # Define the inverse lifting scheme process for one scale (reconstruction)
    def inverse_lifting_scheme(self, s, d, memory, mask, memory_mask):
        # Inverse Update Step - Reconstruct even part from smooth (s) and detail (d)
        even = s - self.unet(d)

        # Inverse Predict Step - Reconstruct odd part from detail (d) and even part
        odd = d + self.pnet(even)

        # Merge the even and odd parts
        # B, T, C = even.shape
        # x = torch.empty(B, T * 2, C, dtype=even.dtype, device=even.device)
        # indices = torch.arange(T * 2, device=even.device).reshape(1, T * 2, 1).expand(B, T * 2, C)
        # even_indices = indices[:, ::2]  # Target positions for even tensor
        # odd_indices = indices[:, 1::2]  # Target positions for odd tensor
        # x = x.scatter(1, even_indices, even)
        # x = x.scatter(1, odd_indices, odd)

        # x = self.s_fuse_op(even, odd)   # even + attn(q=even, v=odd)
        x = torch.mean(torch.stack([even, odd], dim=2), dim=2)
        x = self.s_fuse_op(
            tgt=x,
            memory=memory,
            tgt_mask=mask,
            memory_mask=memory_mask)

        return x

    def forward(self, x, memory, mask, memory_mask):
        if self.block_type == 'analysis':
            s, d = self.lifting_scheme(x, memory, mask, memory_mask)

            return s, d
        elif self.block_type == 'synthesis':
            s, d = x
            x = self.inverse_lifting_scheme(s, d, memory, mask, memory_mask)

            return x
        else:
            raise ValueError(f'Unknown block type: {self.block_type}')



class WaveletNetForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0,
            scaler: str='attn',
            converter: str='id',
            kernel_width: int=3,
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if T_cond > 0:  # True
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:   # False
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # todo
            # # decoder: self-attn, multi-head-attn and feedforward network
            # decoder_layer = nn.TransformerDecoderLayer(
            #     d_model=n_emb,
            #     nhead=n_head,
            #     dim_feedforward=4*n_emb,
            #     dropout=p_drop_attn,
            #     activation='gelu',
            #     batch_first=True,
            #     norm_first=True # important for stability
            # )
            # # a stack of N decoder layers
            # self.decoder = nn.TransformerDecoder(
            #     decoder_layer=decoder_layer,
            #     num_layers=n_layer
            # )

            n_wavelet_layers = math.floor(n_layer / 2.0) + 1

            if converter == 'id':
                self.converter_blocks = nn.Sequential(*[nn.Identity() for _ in range(n_wavelet_layers + 1)])
            elif converter == 'mlp':
                # self.converter_blocks = nn.Sequential(*[MLPBlock(n_emb, n_emb, resid_pdrop=p_drop_attn)
                #                                         for _ in range(config.n_layer + 1)])
                raise NotImplementedError
            elif converter == 'conv':
                self.converter_blocks = nn.Sequential(*[CausalConvBlock(n_emb, n_emb, kernel_size=kernel_width,
                                                                        stride=1, resid_pdrop=p_drop_attn)
                                                        for _ in range(n_wavelet_layers + 1)])
            elif converter == 'attn':
                # self.converter_blocks = nn.Sequential(*[AttentionBlock(config)
                #                                         for _ in range(config.n_layer + 1)])
                raise NotImplementedError
            else:
                raise ValueError(f'Unknown converter: {converter}')

            # wavelet
            self.wavelet_analysis_blocks = nn.Sequential(
                *[WaveletBlock(n_emb, n_emb, horizon, kernel_width, p_drop_attn, 'analysis', scaler=scaler)
                  for _ in range(n_wavelet_layers)])
            self.wavelet_synthesis_blocks = nn.Sequential(
                *[WaveletBlock(n_emb, n_emb, horizon, kernel_width, p_drop_attn, 'synthesis', scaler=scaler)
                  for _ in range(n_wavelet_layers)])

        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn: # True
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def multi_scale_lifting(self, x, memory, mask, memory_mask, scales):
        smooth = x
        details = []

        for scale in range(scales):
            s, d = self.wavelet_analysis_blocks[scale](smooth, memory, mask, memory_mask)
            details.append(d)
            smooth = s  # Continue processing the smooth part

        return smooth, details

    def multi_scale_inverse_lifting(self, smooth, details, memory, mask, memory_mask):
        scales = len(details)

        smoothes = [smooth]
        # Start reconstruction from the final smooth coefficients
        for scale in reversed(range(scales)):
            d = details[scale]
            smooth = self.wavelet_synthesis_blocks[scale]((smooth, d), memory, mask, memory_mask)
            smoothes.append(smooth)

        return smooth, smoothes

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            CausalConvBlock,
            PNet,
            UNet,
            WaveletBlock,
            Functional2Layer,
            nn.Identity)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, WaveletNetForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
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
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.Conv1d, torch.nn.ConvTranspose1d, CausalConv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

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
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )

        return optimizer

    def forward(self, 
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim=20)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim=59)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)  # (B,T,D)

        if self.encoder_only:   # False
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:    # True
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)   # (B,T,n_emb)

            # todo
            # x = self.decoder(
            #     tgt=x,
            #     memory=memory,
            #     tgt_mask=self.mask,
            #     memory_mask=self.memory_mask
            # )
            # (B,T,n_emb)

            scales = len(self.wavelet_analysis_blocks)

            smooth, details = self.multi_scale_lifting(x, memory, self.mask, self.memory_mask, scales)

            # Converter
            smooth = self.converter_blocks[-1](smooth)

            # todo
            for i in range(scales):
                detail = self.converter_blocks[i](details[i])
                details[i] = detail

            merged, smoothes = self.multi_scale_inverse_lifting(smooth, details, memory, self.mask, self.memory_mask)
            x = merged
        
        # head
        x = self.ln_f(x)
        x = self.head(x)    # (B,T,n_out=20)

        d_loss = torch.mean(torch.stack([F.smooth_l1_loss(detail, torch.zeros_like(detail)) for detail in details]))
        s_loss = torch.mean(torch.stack([F.smooth_l1_loss(s_prev, causal_moving_average(s_next, dim=1))
                                         for s_prev, s_next in zip(smoothes[:-1], smoothes[1:])]))

        return x, d_loss, s_loss


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)

