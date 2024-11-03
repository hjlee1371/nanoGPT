from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from llama import Llama, LlamaConfig, RMSNorm, apply_rotary_emb, precompute_freqs_cis

@dataclass
class DeepseekV2Config(LlamaConfig):
    q_lora_rank: Optional[int] = 1536
    kv_lora_rank: int = 512
    nope_head_dim: int = 128
    rope_head_dim: int = 64


class MultiheadLatentAttention(nn.Module):
    def __init__(self, args: DeepseekV2Config):
        super().__init__()
        self.n_heads = args.n_heads
        assert args.n_kv_heads is None, "MultiheadLatentAttention does not support MQA/GQA"
        assert args.qk_norm, "MultiheadLatentAttention requires qk normalization"
        self.head_dim = args.nope_head_dim + args.rope_head_dim
        self.nope_head_dim = args.nope_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank

        if self.q_lora_rank is not None:
            self.wd_q = nn.Linear(
                args.dim,
                args.q_lora_rank,
                bias=False,
            )
            self.wu_q = nn.Linear(
                args.q_lora_rank,
                self.n_heads * self.head_dim,
                bias=False,
            )
            self.q_norm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        else:
            self.wq = nn.Linear(
                args.dim,
                self.n_heads * self.head_dim,
                bias=False,
            )
        self.wd_kv = nn.Linear(
            args.dim,
            self.kv_lora_rank + self.rope_head_dim,
            bias=False,
        )
        self.wu_kv = nn.Linear(
            args.kv_lora_rank,
            self.n_heads * (self.nope_head_dim * 2),
            bias=False,
        )
        self.wo = nn.Linear(
            self.n_heads * self.nope_head_dim,
            args.dim,
            bias=False,
        )

        self.kv_norm = RMSNorm(args.kv_lora_rank, eps=args.norm_eps)


    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        if self.q_lora_rank is not None:
            cq = self.wd_q(x)
            xq = self.wu_q(self.q_norm(cq))
        else:
            xq = self.wq(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq_nope, xq_rope = torch.split(
            xq,
            [self.nope_head_dim, self.rope_head_dim],
            dim=-1,
        )

        ckv, xk_rope = torch.split(
            self.wd_kv(x),
            [self.kv_lora_rank, self.rope_head_dim],
            dim=-1,
        )
        xk_rope = xk_rope.view(bsz, seqlen, 1, self.rope_head_dim)
        xk_nope, xv = torch.split(
            self.wu_kv(self.kv_norm(ckv)).view(bsz, seqlen, self.n_heads, self.nope_head_dim * 2),
            [self.nope_head_dim, self.nope_head_dim],
            dim=-1,
        )

        xq_rope, xk_rope = apply_rotary_emb(xq_rope, xk_rope, freqs_cis=freqs_cis)
        xk_rope = torch.repeat_interleave(xk_rope, dim=2, repeats=self.n_heads)
        xq = torch.cat([xq_nope, xq_rope], dim=-1)
        xk = torch.cat([xk_nope, xk_rope], dim=-1)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class DeepseekV2(Llama):
    def __init__(self, params: DeepseekV2Config):
        super().__init__(params)
        for l in self.layers:
            l.attention = MultiheadLatentAttention(params)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.rope_head_dim, self.params.max_seq_len * 2
        )
        self.apply(self._init_weights)
