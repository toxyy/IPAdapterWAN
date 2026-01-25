"""
Joint Block IP-Adapter Integration Module

Architectural Philosophy:
- Minimal invasive patching of existing attention mechanisms
- Adaptive layer normalization for timestep-conditioned IP injection
- Memory-efficient attention fusion via key-value concatenation

Computational Complexity Analysis:
- IPAttnProcessor.forward: O(seq_len * (img_seq + ip_seq) * dim / heads)
- Memory: O(batch * heads * seq_len * (img_seq + ip_seq)) for attention weights

Design Rationale:
The IP-Adapter pattern injects image conditioning by extending the key-value
space of existing attention layers, preserving the original attention structure
while enabling cross-modal information flow. This is computationally cheaper
than parallel attention streams and maintains gradient flow stability.
"""

from typing import Dict, Optional, Tuple, Any

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import RMSNorm, JointBlock


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with timestep conditioning.
    
    Implements two modes:
    - 'normal': Standard adaLN with shift/scale modulation
    - 'zero': adaLN-Zero with additional gating for residual paths
    
    Mathematical Formulation:
        normal: y = LayerNorm(x) * (1 + scale) + shift
        zero: y = LayerNorm(x) * (1 + scale) + shift, with separate gates
    
    Computational Complexity: O(embedding_dim) per token
    Memory: O(num_params * embedding_dim) for linear projection
    
    Args:
        embedding_dim: Feature dimension for normalization
        time_embedding_dim: Dimension of timestep embedding (defaults to embedding_dim)
        mode: 'normal' (2 params) or 'zero' (6 params for gated residuals)
    """
    
    _MODE_PARAMS = {"zero": 6, "normal": 2}
    
    def __init__(
        self,
        embedding_dim: int,
        time_embedding_dim: Optional[int] = None,
        mode: str = "normal",
    ):
        super().__init__()
        
        if mode not in self._MODE_PARAMS:
            raise ValueError(f"mode must be one of {list(self._MODE_PARAMS.keys())}")
        
        self.mode = mode
        num_params = self._MODE_PARAMS[mode]
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            time_embedding_dim or embedding_dim,
            num_params * embedding_dim,
            bias=True,
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Apply adaptive normalization conditioned on timestep embedding.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)
            emb: Timestep embedding of shape (batch, time_embedding_dim)
            hidden_dtype: Optional dtype override (unused, kept for API compatibility)
        
        Returns:
            normal mode: Normalized tensor of shape (batch, seq_len, embedding_dim)
            zero mode: Tuple of (normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        modulation = self.linear(self.silu(emb))
        
        if self.mode == "normal":
            shift, scale = modulation.chunk(2, dim=1)
            return self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        
        # zero mode: full adaLN-Zero with gating
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
        normalized = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


class IPAttnProcessor(nn.Module):
    """
    IP-Adapter Attention Processor for cross-modal conditioning injection.
    
    Architecture Design:
    - Extends attention key-value space with IP-conditioned projections
    - Uses RMSNorm for query/key normalization (more stable than LayerNorm)
    - Adaptive normalization on IP hidden states for timestep awareness
    
    Attention Pattern:
        Q: image queries (normalized)
        K: concat(image_keys, ip_keys) (both normalized)
        V: concat(image_values, ip_values)
        
    This enables the model to attend to both original context and IP conditioning
    in a single attention operation, avoiding the overhead of separate attention passes.
    
    Computational Complexity: O(seq_len * (img_seq + ip_seq) * dim)
    Memory Overhead: O(ip_seq * hidden_size) for K/V projections
    
    Args:
        hidden_size: Output dimension of attention
        cross_attention_dim: Dimension of cross-attention context (unused, kept for compatibility)
        ip_hidden_states_dim: Dimension of IP hidden states
        ip_encoder_hidden_states_dim: Dimension of IP encoder output (unused)
        head_dim: Dimension per attention head
        timesteps_emb_dim: Dimension of timestep embedding for adaLN
    """
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        ip_hidden_states_dim: int,
        ip_encoder_hidden_states_dim: int,
        head_dim: int,
        timesteps_emb_dim: int = 1280,
    ):
        super().__init__()
        
        # Adaptive normalization for timestep-conditioned IP features
        self.norm_ip = AdaLayerNorm(
            ip_hidden_states_dim,
            time_embedding_dim=timesteps_emb_dim,
            mode="normal",
        )
        
        # IP-specific key-value projections
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        
        # RMSNorm for attention stability (empirically better than LayerNorm for QK)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)
        self.norm_ip_k = RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        ip_hidden_states: Optional[torch.Tensor],
        img_query: torch.Tensor,
        img_key: torch.Tensor,
        img_value: torch.Tensor,
        t_emb: torch.Tensor,
        n_heads: int,
    ) -> Optional[torch.Tensor]:
        """
        Compute IP-augmented attention output.
        
        Args:
            ip_hidden_states: IP features of shape (batch, ip_seq, ip_dim), or None to skip
            img_query: Image queries of shape (batch, seq_len, hidden_size)
            img_key: Image keys of shape (batch, seq_len, hidden_size)
            img_value: Image values of shape (batch, seq_len, n_heads, head_dim)
            t_emb: Timestep embedding of shape (batch, timesteps_emb_dim)
            n_heads: Number of attention heads
        
        Returns:
            Attention output of shape (batch, seq_len, hidden_size), or None if skipped
        
        Note:
            img_value has a different layout (b, l, h, d) vs queries/keys (b, l, hd),
            requiring transpose before concatenation with IP values.
        """
        if ip_hidden_states is None:
            return None
        
        # Normalize IP hidden states with timestep conditioning
        norm_ip_hidden_states = self.norm_ip(ip_hidden_states, emb=t_emb)
        
        # Project to key-value space
        ip_key = self.to_k_ip(norm_ip_hidden_states)
        ip_value = self.to_v_ip(norm_ip_hidden_states)
        
        # Reshape for multi-head attention: (batch, seq, heads*dim) -> (batch, heads, seq, dim)
        img_query = rearrange(img_query, "b l (h d) -> b h l d", h=n_heads)
        img_key = rearrange(img_key, "b l (h d) -> b h l d", h=n_heads)
        ip_key = rearrange(ip_key, "b l (h d) -> b h l d", h=n_heads)
        ip_value = rearrange(ip_value, "b l (h d) -> b h l d", h=n_heads)
        
        # Handle img_value's different layout: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        img_value = img_value.transpose(1, 2)
        
        # Apply RMSNorm to queries and keys for attention stability
        # This prevents attention logit explosion in deep networks
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        ip_key = self.norm_ip_k(ip_key)
        
        # Concatenate image and IP key-values for joint attention
        key = torch.cat([img_key, ip_key], dim=2)
        value = torch.cat([img_value, ip_value], dim=2)
        
        # Scaled dot-product attention with hardware acceleration
        out = F.scaled_dot_product_attention(
            img_query, key, value, dropout_p=0.0, is_causal=False
        )
        
        # Reshape back: (batch, heads, seq, dim) -> (batch, seq, heads*dim)
        out = rearrange(out, "b h l d -> b l (h d)")
        return out.to(img_query.dtype)


class JointBlockIPWrapper:
    """
    Wrapper for JointBlock that injects IP-Adapter conditioning.
    
    Design Pattern: Decorator/Proxy pattern for non-invasive model patching.
    
    This wrapper intercepts the attention computation in JointBlock and augments
    the image branch attention with IP-Adapter features. The context (text) branch
    remains unmodified, preserving the original text conditioning behavior.
    
    Integration Point:
    Used with ComfyUI's set_model_patch_replace to replace attention blocks
    without modifying the original model weights or structure.
    
    Computational Overhead:
    - Additional O(seq_len * ip_seq * dim) for IP attention
    - Memory: O(ip_seq * hidden_size) for IP key-value tensors
    
    Args:
        original_block: The JointBlock instance to wrap
        adapter: IPAttnProcessor for computing IP attention
        ip_options: Dict containing 'hidden_states', 't_emb', and 'weight'
    """
    
    def __init__(
        self,
        original_block: JointBlock,
        adapter: IPAttnProcessor,
        ip_options: Optional[Dict[str, Any]] = None,
    ):
        self.original_block = original_block
        self.adapter = adapter
        self.ip_options = ip_options or {}

    def _block_mixing(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        context_block,
        x_block,
        c: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Execute joint attention with IP-Adapter injection.
        
        This method replicates the logic from mmdit.py's block_mixing but adds
        IP-Adapter attention to the image (x) branch.
        
        Flow:
        1. Compute QKV for both context and x branches
        2. Concatenate and compute joint attention
        3. Split attention outputs
        4. Add IP-Adapter attention to x branch (if enabled)
        5. Apply post-attention processing
        
        Args:
            context: Text/context features of shape (batch, ctx_seq, dim)
            x: Image features of shape (batch, img_seq, dim)
            context_block: Context attention block
            x_block: Image attention block
            c: Conditioning vector
        
        Returns:
            Tuple of (processed_context, processed_x)
        """
        # Pre-attention: compute QKV projections
        context_qkv, context_intermediates = context_block.pre_attention(context, c)
        
        if x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = x_block.pre_attention(x, c)
        
        # Joint attention: concatenate context and image QKV
        qkv = tuple(
            torch.cat((context_qkv[j], x_qkv[j]), dim=1)
            for j in range(3)
        )
        
        attn = optimized_attention(
            qkv[0], qkv[1], qkv[2],
            heads=x_block.attn.num_heads,
        )
        
        # Split attention output back to context and image
        ctx_len = context_qkv[0].shape[1]
        context_attn, x_attn = attn[:, :ctx_len], attn[:, ctx_len:]
        
        # IP-Adapter injection (only when enabled for current timestep)
        ip_hidden = self.ip_options.get("hidden_states")
        t_emb = self.ip_options.get("t_emb")
        
        if ip_hidden is not None and t_emb is not None:
            ip_attn = self.adapter(
                ip_hidden,
                *x_qkv,
                t_emb,
                x_block.attn.num_heads,
            )
            if ip_attn is not None:
                weight = self.ip_options.get("weight", 1.0)
                x_attn = x_attn + ip_attn * weight
        
        # Post-attention processing
        if not context_block.pre_only:
            context = context_block.post_attention(context_attn, *context_intermediates)
        else:
            context = None
        
        if x_block.x_block_self_attn:
            attn2 = optimized_attention(
                x_qkv2[0], x_qkv2[1], x_qkv2[2],
                heads=x_block.attn2.num_heads,
            )
            x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
        else:
            x = x_block.post_attention(x_attn, *x_intermediates)
        
        return context, x

    def __call__(
        self,
        args: Dict[str, torch.Tensor],
        _: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Callable interface for ComfyUI's patch_replace mechanism.
        
        Args:
            args: Dict with 'txt' (context), 'img' (image features), 'vec' (conditioning)
            _: Unused second argument (original block wrapper, not accessible)
        
        Returns:
            Dict with processed 'txt' and 'img' tensors
        """
        context, x = self._block_mixing(
            args["txt"],
            args["img"],
            self.original_block.context_block,
            self.original_block.x_block,
            c=args["vec"],
        )
        return {"txt": context, "img": x}
