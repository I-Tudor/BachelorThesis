"""
rna_transformer.py

Improved RNATransformer for the PARC audio dataset.

Incorporated improvements
(1) Per-stream 1-D ConvNet front-end
    AugmentedNet (Nápoles López et al., ISMIR 2021)

(2) Rotary Positional Encoding (RoPE) inside attention
    Su et al., "RoFormer", 2021 - arXiv:2104.09864

(3) Token Merging (ToMe) with full unmerge for token-level prediction
    Bolya et al., "Token Merging: Your ViT but Faster", ICLR 2023
    arXiv:2210.09461

(4) FiLM-based key conditioning applied to ALL task heads
    RNBert (Sailor, ISMIR 2024) + FiLM (Perez et al., AAAI 2018)

(5) Chord-change auxiliary detection head
    Harmony Transformer (Chen & Su, TISMIR 2021)

(6) Homoscedastic uncertainty-weighted multi-task loss
    Kendall et al., "Multi-Task Learning Using Uncertainty…", CVPR 2018

(7) [NEW] CLS token for global-key representation
    Prepended to the sequence; its output feeds the key head instead
    of mean-pooling, giving the model a dedicated slot to accumulate
    global tonal context across all attention layers.

(8) [NEW] Pitch-class profile (PCP) shortcut to key head
    Raw mean chroma + bass-chroma vectors are concatenated with the
    CLS output before the key prediction head, providing the same
    tonal-histogram signal that classical key-finding algorithms use.

"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Internal helpers

def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest divisor of *channels* that is ≤ max_groups (minimum 1)."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# 1.  Per-stream 1-D convolutional front-end

class ConvStream(nn.Module):
    """
    Multi-scale 1-D convolutional feature extractor for one input stream.

    Four blocks with kernel sizes (1, 3, 7, 15) give progressively wider
    local receptive fields before global self-attention kicks in, mirroring
    the per-stream conv architecture of AugmentedNet (Nápoles López et al.,
    ISMIR 2021).  A residual shortcut is added for gradient flow.

    Shape
    Input  : [B, C_in, T]   - channels-first (DataLoader format)
    Output : [B, d_out,  T]
    """

    _KERNELS: Tuple[int, ...] = (1, 3, 7, 15)

    def __init__(self, c_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(d_out * 2, c_in * 4, 64)

        blocks: List[nn.Module] = []
        ch = c_in
        for i, k in enumerate(self._KERNELS):
            out_ch = d_out if i == len(self._KERNELS) - 1 else hidden
            blocks.append(nn.Sequential(
                nn.Conv1d(ch, out_ch, kernel_size=k,
                          padding=(k - 1) // 2, bias=False),
                nn.GroupNorm(_num_groups(out_ch), out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            ch = out_ch

        self.blocks   = nn.ModuleList(blocks)
        self.shortcut = (
            nn.Conv1d(c_in, d_out, 1, bias=False)
            if c_in != d_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        for block in self.blocks:
            x = block(x)
        return x + residual                                   # [B, d_out, T]


# 2.  Rotary Positional Encoding (RoPE)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate each (x1, x2) pair -> (−x2, x1) along the last dimension."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding - Su et al. (2021), arXiv:2104.09864.

    Applied to Q and K inside attention so position information is relative
    rather than absolute, which generalises better to variable-length pieces.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (
            10_000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    @torch.no_grad()
    def _build_cache(self, seq_len: int) -> None:
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)               # [T, hd/2]
        emb   = torch.cat([freqs, freqs], dim=-1)           # [T, hd]
        # Shape [1, 1, T, hd] for broadcasting over batch and heads
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k : [B, nhead, T, head_dim]
        Returns rotated (q_rot, k_rot) of the same shape.
        """
        T = q.size(2)
        if T > self._cos.size(2):
            self._build_cache(T * 2)
        cos = self._cos[:, :, :T]                           # [1, 1, T, hd]
        sin = self._sin[:, :, :T]
        return (q * cos + _rotate_half(q) * sin,
                k * cos + _rotate_half(k) * sin)


# 3.  Token Merging (ToMe) - Bolya et al., ICLR 2023

@dataclass
class ToMeState:
    """
    Everything needed to undo a single round of token merging.

    Attributes
    ──────────
    T_orig   : original sequence length before merge
    n_dst    : number of destination tokens (even-index tokens)
    src_idx  : [B, r]          – src slots that were merged
    unm_idx  : [B, n_src − r]  – src slots that were kept
    dst_idx  : [B, r]          – dst slot each merged src went into
    dst_orig : [n_dst]         – original positions of dst tokens  (0,2,4,…)
    src_orig : [n_src]         – original positions of src tokens  (1,3,5,…)
    """
    T_orig  : int
    n_dst   : int
    src_idx : torch.Tensor
    unm_idx : torch.Tensor
    dst_idx : torch.Tensor
    dst_orig: torch.Tensor
    src_orig: torch.Tensor


@torch.no_grad()
def tome_match(metric: torch.Tensor, r: int) -> ToMeState:
    """
    Bipartite soft matching from ToMe (Bolya et al., ICLR 2023).

    Partitions the sequence into *dst* (even indices) and *src* (odd indices),
    ranks all (src, dst) pairs by cosine similarity of the supplied metric
    vectors, and returns a ToMeState for the top-r pairs.

    Parameters
    metric : [B, T, C]   – key vectors from self-attention (post-RoPE)
    r      : int         – pairs to merge (clipped to min(r, n_src))

    Returns
    ToMeState
    """
    B, T, _ = metric.shape
    dst_orig = torch.arange(0, T, 2, device=metric.device)  # even
    src_orig = torch.arange(1, T, 2, device=metric.device)  # odd
    n_dst, n_src = dst_orig.numel(), src_orig.numel()
    r = min(r, n_src)

    dst_m = metric[:, 0::2, :]                               # [B, n_dst, C]
    src_m = metric[:, 1::2, :]                               # [B, n_src, C]

    sim = torch.bmm(
        F.normalize(src_m, dim=-1),
        F.normalize(dst_m, dim=-1).transpose(1, 2),
    )                                                        # [B, n_src, n_dst]

    scores, node_max = sim.max(dim=-1)                       # [B, n_src]
    order   = scores.argsort(dim=-1, descending=True)
    src_idx = order[:, :r]                                   # [B, r]  - merged
    unm_idx = order[:, r:]                                   # [B, n_src−r]
    dst_idx = node_max.gather(1, src_idx)                    # [B, r]

    return ToMeState(T, n_dst, src_idx, unm_idx, dst_idx, dst_orig, src_orig)


def tome_merge(x: torch.Tensor, state: ToMeState) -> torch.Tensor:
    """
    Merge top-r src tokens into their matched dst tokens (size-weighted avg),
    then concatenate the unmatched src tokens.

    Input  : [B, T,   D]
    Output : [B, T−r, D]
    """
    B, _, D = x.shape
    r = state.src_idx.shape[1]

    dst = x[:, 0::2, :]                                      # [B, n_dst, D]
    src = x[:, 1::2, :]                                      # [B, n_src, D]

    # Accumulate merged src tokens into matched dst slots
    idx      = state.dst_idx.unsqueeze(-1).expand(-1, -1, D) # [B, r, D]
    src_sel  = src.gather(
        1, state.src_idx.unsqueeze(-1).expand(-1, -1, D))    # [B, r, D]

    # Out-of-place scatter_add so autograd can track both src and dst paths
    merged_dst = dst.clone().scatter_add_(1, idx, src_sel)   # [B, n_dst, D]

    # Normalise by the count of tokens that landed in each dst slot
    count = torch.ones(B, state.n_dst, 1, device=x.device, dtype=x.dtype)
    count.scatter_add_(1, state.dst_idx.unsqueeze(-1),
                       torch.ones(B, r, 1, device=x.device, dtype=x.dtype))
    merged_dst = merged_dst / count                          # [B, n_dst, D]

    # Unmatched src tokens pass through unchanged
    unm = src.gather(
        1, state.unm_idx.unsqueeze(-1).expand(-1, -1, D))   # [B, n_src−r, D]

    return torch.cat([merged_dst, unm], dim=1)               # [B, T−r, D]


def tome_unmerge(x: torch.Tensor, state: ToMeState) -> torch.Tensor:
    """
    Restore original sequence length after a round of merging.

    Merged src positions receive a copy of their matched dst token's output;
    unmerged src positions are placed back where they started.

    Input  : [B, T−r, D]
    Output : [B, T,   D]
    """
    B, _, D = x.shape
    T = state.T_orig

    dst_out = x[:, :state.n_dst, :]                          # [B, n_dst,   D]
    unm_out = x[:, state.n_dst:, :]                          # [B, n_src−r, D]

    unm_orig = state.src_orig[state.unm_idx]                 # [B, n_src−r]
    mrg_orig = state.src_orig[state.src_idx]                 # [B, r]
    mrg_val  = dst_out.gather(
        1, state.dst_idx.unsqueeze(-1).expand(-1, -1, D))   # [B, r, D]

    all_idx = torch.cat([
        state.dst_orig.unsqueeze(0).expand(B, -1),          # [B, n_dst]
        unm_orig,                                            # [B, n_src−r]
        mrg_orig,                                            # [B, r]
    ], dim=1)                                                # [B, T]

    all_val = torch.cat([dst_out, unm_out, mrg_val], dim=1) # [B, T, D]

    out = torch.zeros(B, T, D, device=x.device, dtype=x.dtype).scatter(
        1,
        all_idx.unsqueeze(-1).expand(-1, -1, D),
        all_val,
    )
    return out                                               # [B, T, D]


# 4.  Custom encoder layer - RoPE + Flash Attention + optional ToMe

class RNAEncoderLayer(nn.Module):
    """
    Pre-LN Transformer encoder layer with:

    - RoPE on Q and K (relative position, better length generalisation)
    - F.scaled_dot_product_attention (uses Flash Attention when available)
    - Optional Token Merging (tome_r > 0)

    """

    def __init__(
        self,
        d_model        : int,
        nhead          : int,
        dim_feedforward: int   = 1024,
        dropout        : float = 0.1,
        tome_r         : int   = 0,
        max_seq_len    : int   = 512,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_model  = d_model
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.tome_r   = tome_r
        self.drop_p   = dropout

        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def _to_heads(self, t: torch.Tensor, T: int) -> torch.Tensor:
        B = t.shape[0]
        return t.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def _from_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, _, Tp, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, Tp, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[ToMeState]]:
        """
        x : [B, T, d_model]

        Returns
        output    : [B, T′, d_model]   T′ = T − r  (or T if tome_r = 0)
        tome_state: ToMeState / None
        """
        B, T, _ = x.shape

        h   = self.norm1(x)
        qkv = self.qkv_proj(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._to_heads(q, T)
        k = self._to_heads(k, T)
        v = self._to_heads(v, T)

        q, k = self.rope(q, k)

        state: Optional[ToMeState] = None
        if self.tome_r > 0:
            metric = self._from_heads(k)
            state  = tome_match(metric, self.tome_r)

            q = tome_merge(self._from_heads(q), state)
            k = tome_merge(self._from_heads(k), state)
            v = tome_merge(self._from_heads(v), state)
            Tp = q.shape[1]
            q  = self._to_heads(q, Tp)
            k  = self._to_heads(k, Tp)
            v  = self._to_heads(v, Tp)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop_p if self.training else 0.0,
        )
        attn_out = self.out_proj(self._from_heads(attn_out))

        x_res = tome_merge(x, state) if state is not None else x
        x = x_res + attn_out

        x = x + self.ffn(self.norm2(x))

        return x, state


# 5.  FiLM key conditioner

class FiLMKeyConditioner(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on key predictions.

    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018, extended to condition ALL task heads on global-key logits
    (following the spirit of RNBert, Sailor, ISMIR 2024).

    Given key logits -> softmax probs -> small MLP -> (γ, β):
        h_out = γ ⊙ h + β

    Key logits are detached so that gradients from chord / inversion heads
    cannot distort key-head training through the conditioning path.
    """

    def __init__(self, key_classes: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        hidden = max(d_model // 4, key_classes * 2)
        self.net = nn.Sequential(
            nn.Linear(key_classes, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        self._reset_film_bias()

    def _reset_film_bias(self) -> None:
        """Start training with identity transform: γ = 1, β = 0."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.ones_(self.net[-1].bias[:self.d_model])
        nn.init.zeros_(self.net[-1].bias[self.d_model:])

    def forward(self, h: torch.Tensor, key_logits: torch.Tensor) -> torch.Tensor:
        """
        h         : [B, T, d_model]
        key_logits: [B, T, n_key]   - detached inside this method
        Returns   : [B, T, d_model]
        """
        probs        = F.softmax(key_logits.detach(), dim=-1)
        gamma, beta  = self.net(probs).chunk(2, dim=-1)
        return gamma * h + beta


# 6.  Per-task MLP head

class TaskHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(in_dim // 2, out_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 7.  Homoscedastic uncertainty-weighted multi-task loss

class UncertaintyWeightedLoss(nn.Module):
    """
    Learns a per-task log-variance log σ²_k and weighs each task as:

        L_total = Σ_k  [ exp(−log σ²_k) · L_k  +  log σ²_k ]

    Hard tasks naturally get smaller σ²_k (higher weight); easy tasks are
    down-weighted automatically.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty…",
               CVPR 2018.
    """

    def __init__(self, task_names: List[str]) -> None:
        super().__init__()
        self.task_names = task_names
        self.log_sigma2 = nn.ParameterDict({
            name.replace(".", "_"): nn.Parameter(torch.zeros(()))
            for name in task_names
        })

    def forward(
        self,
        per_task_losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(iter(per_task_losses.values())).device
        total  = torch.zeros((), device=device)
        task_weights: Dict[str, float] = {}

        for name in self.task_names:
            if name not in per_task_losses:
                continue
            log_s2 = self.log_sigma2[name.replace(".", "_")]
            prec   = torch.exp(-log_s2)
            total  = total + prec * per_task_losses[name] + log_s2
            task_weights[name] = float(prec.detach())

        return total, task_weights


# 8.  ImprovedRNATransformer - main model

class ImprovedRNATransformer(nn.Module):
    """
    Parameters
    in_channels : tuple[int, …]
        Feature channels per input stream - same semantics as the original
        RNATransformer.  (12, 12) for NNLS-Chroma+Bass; (84,) for semitone.
    label_sizes : dict[str, int]
        Maps task name -> vocabulary size.
    d_model : int            Transformer hidden dimension  (default 256)
    nhead : int              Attention heads               (default 8)
    num_layers : int         Encoder depth                 (default 6)
    dim_feedforward : int    FFN inner dimension           (default 1024)
    dropout : float          Dropout rate                  (default 0.1)
    key_head_dropout : float Dropout for the key head specifically (default 0.35)
                             Higher than the global rate to regularise the
                             key head independently.
    max_seq_len : int        Cache size for RoPE           (default 512)
    tome_r : int
        Tokens merged per encoder layer (ToMe).  Must be 0 when using the
        CLS token (current default).  Set to 0 until CLS + ToMe integration
        is implemented.
    add_chord_change_head : bool
        Automatically inserts a binary chord-change head if "chord_change"
        is not already in label_sizes.
    pcp_channels : int
        Number of channels in the pitch-class profile shortcut fed to the
        key head.  Defaults to 24 (12 chroma + 12 bass-chroma).  Set to 12
        if only one input stream is used (semitone spectrum mode).

    Key-head architecture
    Instead of mean-pooling the encoder output:

        1. A learnable CLS token is prepended to the frame sequence before
           the encoder.  Its final output captures global tonal context
           via self-attention over the entire segment.
        2. The raw mean pitch-class profile (PCP) of the input features is
           concatenated with the CLS output.  This provides the tonal
           histogram directly to the key head, acting as a strong inductive
           bias analogous to the Krumhansl-Schmuckler algorithm.
        3. The combined [CLS_out ‖ PCP] vector (d_model + pcp_channels) is
           passed through a dedicated key-head MLP with higher dropout.

    Output format
    dict[str, Tensor [B, T, n_classes]]
    Identical to the original RNATransformer.
    """

    def __init__(
        self,
        in_channels          : Tuple[int, ...],
        label_sizes          : Dict[str, int],
        d_model              : int   = 256,
        nhead                : int   = 8,
        num_layers           : int   = 6,
        dim_feedforward      : int   = 1024,
        dropout              : float = 0.1,
        key_head_dropout     : float = 0.35,
        max_seq_len          : int   = 512,
        tome_r               : int   = 0,
        add_chord_change_head: bool  = False,
        pcp_channels         : int   = 24,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert tome_r == 0, (
            "tome_r > 0 is not supported when using the CLS token. "
            "Set tome_r=0 or implement CLS-aware ToMe."
        )

        self.label_sizes  = dict(label_sizes)
        self.tome_r       = tome_r
        self.pcp_channels = pcp_channels

        if add_chord_change_head and "chord_change" not in self.label_sizes:
            self.label_sizes["chord_change"] = 2

        # Per-stream conv front-end
        n_streams = len(in_channels)
        d_base    = d_model // n_streams
        d_streams = [d_base] * (n_streams - 1) + [
            d_model - d_base * (n_streams - 1)
        ]

        self.conv_streams = nn.ModuleList([
            ConvStream(c, d_s, dropout=dropout)
            for c, d_s in zip(in_channels, d_streams)
        ])
        self.proj_norm = nn.LayerNorm(d_model)

        # CLS token
        # Learnable token prepended to the frame sequence; its encoder output
        # captures global tonal context and drives key prediction.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            RNAEncoderLayer(
                d_model        = d_model,
                nhead          = nhead,
                dim_feedforward= dim_feedforward,
                dropout        = dropout,
                tome_r         = tome_r,
                max_seq_len    = max_seq_len + 1,  # +1 for CLS position
            )
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # FiLM key conditioner
        self.film = FiLMKeyConditioner(self.label_sizes["global_key"], d_model)

        # Task heads
        # Key head: takes CLS output + PCP shortcut -> wider input
        # All other heads: take d_model frame features
        self.task_heads = nn.ModuleDict()
        for name, n_cls in self.label_sizes.items():
            if name == "global_key":
                self.task_heads[name] = TaskHead(
                    d_model + pcp_channels,
                    n_cls,
                    dropout=key_head_dropout,
                )
            else:
                self.task_heads[name] = TaskHead(d_model, n_cls, dropout=dropout)

        self._init_weights()

    # Weight initialisation

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Re-apply FiLM's careful initialisation (xavier above would overwrite it)
        self.film._reset_film_bias()
        # Re-apply CLS token init (xavier above skips Parameters)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # Forward

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        features : list[Tensor [B, C_i, T]]  - channels-first

        Returns
        dict[str, Tensor [B, T, n_classes]]
        """
        # 1. Per-stream conv front-end  [B, C_i, T] -> [B, d_stream_i, T]
        stream_outs = [conv(f) for conv, f in zip(self.conv_streams, features)]
        x = torch.cat(stream_outs, dim=1)        # [B, d_model, T]
        x = x.permute(0, 2, 1).contiguous()      # [B, T, d_model]
        x = self.proj_norm(x)
        B, T, _ = x.shape

        # 2. Compute mean PCP from raw input features (before the conv)
        #    This is the pitch-class histogram shortcut to the key head.
        #    features[0]: chroma [B, 12, T], features[1]: bass-chroma [B, 12, T]
        mean_chroma = features[0].mean(dim=-1)    # [B, 12]
        if len(features) > 1:
            mean_bass = features[1].mean(dim=-1)  # [B, 12]
        else:
            mean_bass = torch.zeros(B, 12, device=x.device, dtype=x.dtype)

        # Clamp to [0, 1] - chroma values can occasionally be slightly negative
        # due to NNLS estimation; clamp prevents negative histogram values
        # from confusing the key head.
        mean_chroma = mean_chroma.clamp(min=0.0)
        mean_bass   = mean_bass.clamp(min=0.0)
        pcp = torch.cat([mean_chroma, mean_bass], dim=-1)   # [B, pcp_channels]

        # Normalise PCP to sum-to-one (avoids amplitude confound)
        pcp_sum = pcp.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pcp = pcp / pcp_sum                                  # [B, pcp_channels]

        # 3. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x   = torch.cat([cls, x], dim=1)        # [B, T+1, d_model]

        # 4. Encoder - CLS token participates in all attention layers
        #    It attends to all frame tokens and vice versa, letting the
        #    model accumulate global tonal information into the CLS slot.
        tome_states: List[Optional[ToMeState]] = []
        for layer in self.encoder_layers:
            x, state = layer(x)
            tome_states.append(state)
        x = self.encoder_norm(x)                 # [B, T+1, d_model]

        # 5. Split CLS output from frame outputs
        cls_out = x[:, 0, :]                     # [B, d_model]
        x       = x[:, 1:, :]                    # [B, T,   d_model]

        # 6. Reverse Token Merging (no-op when tome_r == 0)
        for state in reversed(tome_states):
            if state is not None:
                x = tome_unmerge(x, state)

        # 7. Predict global_key from CLS output + PCP shortcut
        key_repr       = torch.cat([cls_out, pcp], dim=-1)       # [B, d_model+pcp]
        key_logits_seg = self.task_heads["global_key"](key_repr)  # [B, n_key]
        # Broadcast segment-level key prediction to all frames for FiLM
        key_logits = key_logits_seg.unsqueeze(1).expand(-1, T, -1)  # [B, T, n_key]

        # 8. FiLM: condition all remaining task features on key predictions
        h = self.film(x, key_logits)             # [B, T, d_model]

        # 9. Per-task heads on conditioned features
        outputs: Dict[str, torch.Tensor] = {"global_key": key_logits}
        for name in self.label_sizes:
            if name == "global_key":
                continue
            outputs[name] = self.task_heads[name](h)

        return outputs

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def key_head_parameters(self):
        """Returns parameters belonging specifically to the key head and CLS token.
        Used to set a separate learning rate / weight decay in the optimiser."""
        return (
            list(self.task_heads["global_key"].parameters()) +
            [self.cls_token]
        )

    def non_key_parameters(self):
        """Returns all parameters except the key head and CLS token."""
        key_ids = {id(p) for p in self.key_head_parameters()}
        return [p for p in self.parameters() if id(p) not in key_ids]


# 9.  Utility: derive chord-change targets from existing frame-level labels

def derive_chord_change(
    labels   : torch.Tensor,
    pad_value: int = -1,
) -> torch.Tensor:
    """
    Build binary chord-change targets from any frame-level label sequence.

    A change is flagged at frame t when labels[t] != labels[t − 1].
    Frame 0 is always treated as a change (start of sequence).

    Parameters
    labels    : [B, T]  int64
    pad_value : frames set to this value are excluded

    Returns
    [B, T]  int64  - 1 at chord-change frames, 0 elsewhere
    """
    B, T = labels.shape
    out = torch.zeros(B, T, dtype=torch.long, device=labels.device)
    out[:, 0] = 1
    out[:, 1:] = (labels[:, 1:] != labels[:, :-1]).long()

    if pad_value != -100:
        out[labels == pad_value] = pad_value
    else:
        out[labels == -100] = -100

    return out

RNATransformer = ImprovedRNATransformer