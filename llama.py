# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# originally from https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py
# Modified by Hojin Lee

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import moe_ops

@dataclass
class LlamaConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000 # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    qk_norm: bool = False

    scale_factor: int = 1
    parametrization: str = "sp"

    num_total_experts: int = 1
    num_active_experts: int = 1
    aux_loss_coeff: float = 1e-2 # from switch transformer

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.q_norm = RMSNorm(args.dim, eps=args.norm_eps) if args.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(args.dim, eps=args.norm_eps) if args.qk_norm else nn.Identity()

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        num_total_experts: int,
        num_active_experts: int,
        aux_loss_coeff: float,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # comparable active params for moe & dense with same configurations
        assert hidden_dim % num_active_experts == 0
        hidden_dim = hidden_dim // num_active_experts
        assert hidden_dim % 8 == 0
        self.hidden_dim = hidden_dim
        self.num_total_experts = num_total_experts
        self.num_active_experts = num_active_experts
        self.aux_loss_coeff = aux_loss_coeff

        self.router = nn.Linear(dim, num_total_experts, bias=None)
        self.w1 = nn.Parameter(
            torch.empty(num_total_experts, dim, hidden_dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_total_experts, hidden_dim, dim)
        )
        self.w3 = nn.Parameter(
            torch.empty(num_total_experts, dim, hidden_dim)
        )

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        x = x.view(-1, dim)
        scores = F.softmax(self.router(x), dim=-1)
        expert_weights, expert_indices = torch.topk(scores, self.num_active_experts, dim=-1)
        expert_weights = expert_weights.flatten()
        expert_indices = expert_indices.int().flatten()
        with torch.no_grad():
            bin_ids, indices = torch.sort(expert_indices)
            num_tokens_per_expert = torch.histc(expert_indices, self.num_total_experts)
            bins = torch.cumsum(num_tokens_per_expert, 0)
            bin_ids = bin_ids.int()
            bins = bins.int()

        num_tokens_per_expert = num_tokens_per_expert.cpu().to(torch.long)
        x = moe_ops.gather(x, indices, bin_ids, bins, self.num_active_experts)
        x = F.silu(
            moe_ops.gmm(x, self.w1, num_tokens_per_expert)
        ) * moe_ops.gmm(x, self.w3, num_tokens_per_expert)
        x = moe_ops.gmm(x, self.w2, num_tokens_per_expert)
        x = moe_ops.scatter(x, indices, bin_ids, expert_weights, bins, self.num_active_experts)
        return x.view(bsz, seqlen, dim)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LlamaConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        if args.num_total_experts > 1:
            self.feed_forward = MoEFeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                num_total_experts=args.num_total_experts,
                num_active_experts=args.num_active_experts,
                aux_loss_coeff=args.aux_loss_coeff,
            )
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama(nn.Module):
    def __init__(self, params: LlamaConfig):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.dim = params.dim
        self.n_heads = params.n_heads
        self.n_kv_heads = params.n_kv_heads if params.n_kv_heads is not None else params.n_heads
        hidden_dim = int(8 * self.dim / 3)
        if params.ffn_dim_multiplier is not None:
            hidden_dim = int(params.ffn_dim_multiplier * hidden_dim)
        self.ffn_hidden_dim = params.multiple_of * ((hidden_dim + params.multiple_of - 1) // params.multiple_of)

        self.scale_factor = params.scale_factor
        self.parametrization = params.parametrization

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            assert module.weight.ndim == 2
            assert module.bias is None
            fan_in = module.weight.size(1)
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=1./math.sqrt(fan_in))
        if isinstance(module, MoEFeedForward):
            assert module.w1.ndim == 3
            assert module.w2.ndim == 3
            assert module.w3.ndim == 3
            _, dim, hidden_dim = module.w1.shape
            torch.nn.init.trunc_normal_(module.w1, mean=0.0, std=1./math.sqrt(dim))
            torch.nn.init.trunc_normal_(module.w3, mean=0.0, std=1./math.sqrt(dim))
            torch.nn.init.trunc_normal_(module.w2, mean=0.0, std=1./math.sqrt(hidden_dim))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1./math.sqrt(self.dim))

    def forward(self, tokens, targets, z_loss_coeff):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        start_pos = 0

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis)
        h = self.norm(h)
        output = self.output(h).float()
        ntp_loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1), ignore_index=-1)
        z_loss = z_loss_coeff * torch.logsumexp(output, dim=-1).square().mean()
        return output, ntp_loss, z_loss

    def configure_optimizers(self, weight_decay, independent_weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        # some hack to use independent weight decay with pytorch optims
        weight_decay = weight_decay / learning_rate if independent_weight_decay else weight_decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        embed_params = [p for n, p in param_dict.items() if p.dim() >= 2 and n.endswith("tok_embeddings.weight")]
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and not n.endswith("tok_embeddings.weight")]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        if self.parametrization == "sp":
            optim_groups = [
                {
                    'params': embed_params,
                    'weight_decay': weight_decay,
                },
                {
                    'params': decay_params,
                    'weight_decay': weight_decay,
                },
                {
                    'params': nodecay_params,
                    'weight_decay': 0.0,
                }
            ]
        elif self.parametrization == "mup-simple":
            optim_groups = [
                {
                    'params': embed_params,
                    'weight_decay': weight_decay,
                },
                {
                    'params': decay_params,
                    'lr': learning_rate / self.scale_factor,
                    # Weight decay should scale once more for mup w/ independent_weight_decay
                    # See https://github.com/microsoft/mup/issues/1 for details
                    'weight_decay': weight_decay * self.scale_factor if independent_weight_decay else weight_decay,
                },
                {
                    'params': nodecay_params,
                    'weight_decay': 0.0,
                }
            ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, bsz, seqlen, dt):
        # modified from https://github.com/NVIDIA/Megatron-LM/blob/b1218b905bd4bed88c17bcee6ddd9bee0b5b3278/megatron/training/training.py#L98
        # The 12x term below comes from the following factors; for more details, see
        # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
        #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
        #       in MLP layer).
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        gated_linear_multiplier = 3 / 2
        expansion_factor = 3 * 2 * 2
        flops = (
            expansion_factor
            * bsz
            * seqlen
            * self.n_layers
            * self.dim
            * self.dim
            * (
                # Attention.
                (
                    (
                        1
                        + (self.n_kv_heads / self.n_heads)
                        + (seqlen / self.dim)
                    )
                )
                # MLP.
                + (
                    (self.ffn_hidden_dim / self.dim)
                    * gated_linear_multiplier
                )
                # Logit.
                + (self.vocab_size / (2 * self.n_layers * self.dim))
            )
        )
        return flops / 312e12 / dt
