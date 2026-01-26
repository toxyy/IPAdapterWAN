# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# Enhanced with architectural improvements for computational efficiency and maintainability

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import Timesteps, TimestepEmbedding


def FeedForward(dim: int, mult: int = 4) -> nn.Sequential:
    """
    Position-wise Feed-Forward Network with pre-normalization.
    
    Computational Complexity: O(d * d * mult) per token
    Memory: O(d * mult) for intermediate activations
    
    Args:
        dim: Input/output dimension
        mult: Hidden dimension multiplier (default 4x expansion)
    
    Returns:
        Sequential module: LayerNorm -> Linear -> GELU -> Linear
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x: torch.Tensor, heads: int) -> torch.Tensor:
    """
    Reshape tensor for multi-head attention computation.
    
    Memory-efficient reshape avoiding unnecessary copies via view/transpose.
    
    Args:
        x: Input tensor of shape (batch, length, width)
        heads: Number of attention heads
    
    Returns:
        Reshaped tensor of shape (batch, heads, length, dim_per_head)
    """
    bs, length, width = x.shape
    # Fused reshape: (bs, length, width) -> (bs, heads, length, dim_per_head)
    return x.view(bs, length, heads, -1).transpose(1, 2).contiguous()


class PerceiverAttention(nn.Module):
    """
    Cross-attention module implementing Perceiver-style latent attention.
    
    Architecture: Queries from latents, Keys/Values from concatenated (input, latents)
    This enables information flow from high-dimensional input to compressed latent space.
    
    Computational Complexity: O(n_latents * (n_input + n_latents) * d)
    Memory: O(n_latents * (n_input + n_latents)) for attention weights
    """
    
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional adaptive layer normalization.
        
        Args:
            x: Image features of shape (batch, n_tokens, dim)
            latents: Latent features of shape (batch, n_queries, dim)
            shift: Optional shift for adaLN of shape (batch, dim)
            scale: Optional scale for adaLN of shape (batch, dim)
        
        Returns:
            Updated latents of shape (batch, n_queries, dim)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        # Adaptive Layer Normalization: latents = latents * (1 + scale) + shift
        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Scaled dot-product attention with numerical stability
        # Using F.scaled_dot_product_attention for hardware acceleration when available
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape back: (batch, heads, length, dim_head) -> (batch, length, dim)
        out = out.transpose(1, 2).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    """
    Perceiver-based resampler for compressing variable-length sequences to fixed queries.
    
    Design Philosophy: Cross-attention bottleneck that learns to extract relevant
    information from arbitrary input sequences into a fixed number of latent queries.
    
    Use Case: Image feature compression for downstream conditioning.
    """
    
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        embedding_dim: int = 768,
        output_dim: int = 1024,
        ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Learnable latent queries with scaled initialization for stability
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compress input features to latent queries.
        
        Args:
            x: Input features of shape (batch, seq_len, embedding_dim)
        
        Returns:
            Compressed features of shape (batch, num_queries, output_dim)
        """
        latents = self.latents.expand(x.size(0), -1, -1)
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class TimeResampler(nn.Module):
    """
    Time-conditioned Perceiver resampler for diffusion model conditioning.
    
    Architecture Enhancement: Integrates timestep conditioning via:
    1. Additive timestep embedding to input features
    2. Adaptive Layer Normalization (adaLN) in attention and FFN blocks
    
    This enables the resampler to produce timestep-aware representations,
    critical for diffusion model guidance where conditioning strength
    should vary across the denoising trajectory.
    
    Computational Complexity: O(depth * num_queries * (seq_len + num_queries) * dim)
    Memory: O(num_queries * dim) for latents + O(seq_len * dim) for projected input
    """
    
    def __init__(
        self,
        dim: int = 1024,
        depth: int = 8,
        dim_head: int = 64,
        heads: int = 16,
        num_queries: int = 8,
        embedding_dim: int = 768,
        output_dim: int = 1024,
        ff_mult: int = 4,
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
    ):
        super().__init__()
        
        # Store config for potential serialization/debugging
        self.dim = dim
        self.num_queries = num_queries
        
        # Learnable latent queries with variance-preserving initialization
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)

        # Input/output projections
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        # Transformer layers with adaLN modulation
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
                # adaLN modulation: produces (shift_msa, scale_msa, shift_mlp, scale_mlp)
                nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True)),
            ])
            for _ in range(depth)
        ])

        # Timestep embedding pipeline: sinusoidal -> MLP
        self.time_proj = Timesteps(
            timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift
        )
        self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        need_temb: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with timestep conditioning.
        
        Args:
            x: Input features of shape (batch, seq_len, embedding_dim)
            timestep: Diffusion timestep (scalar or tensor)
            need_temb: If True, also return the timestep embedding
        
        Returns:
            If need_temb=False: latents of shape (batch, num_queries, output_dim)
            If need_temb=True: (latents, timestep_emb) tuple
        """
        timestep_emb = self._embed_timestep(x, timestep)

        # Expand latents for batch
        latents = self.latents.expand(x.size(0), -1, -1)

        # Project input and add timestep conditioning
        x = self.proj_in(x)
        x = x + timestep_emb.unsqueeze(1)

        # Process through transformer layers with adaLN
        for attn, ff, adaLN_modulation in self.layers:
            # Compute modulation parameters from timestep embedding
            modulation = adaLN_modulation(timestep_emb)
            shift_msa, scale_msa, shift_mlp, scale_mlp = modulation.chunk(4, dim=1)
            
            # Attention with adaLN on latents
            latents = attn(x, latents, shift_msa, scale_msa) + latents

            # FFN with adaLN: apply modulation after first LayerNorm
            res = latents
            for idx, layer in enumerate(ff):
                latents = layer(latents)
                # Apply adaLN after the LayerNorm (first layer in FFN)
                if idx == 0 and isinstance(layer, nn.LayerNorm):
                    latents = latents * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            latents = latents + res

        # Output projection and normalization
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        if need_temb:
            return latents, timestep_emb
        return latents

    def _embed_timestep(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        """
        Convert timestep to embedding vector.
        
        Handles various input formats (scalar, tensor) and ensures proper
        dtype/device alignment with the sample tensor.
        
        Args:
            sample: Reference tensor for device/dtype inference
            timestep: Diffusion timestep
        
        Returns:
            Timestep embedding of shape (batch, dim)
        """
        if not torch.is_tensor(timestep):
            # Handle scalar timesteps with appropriate dtype
            is_mps = sample.device.type == "mps"
            dtype = torch.float32 if is_mps else (
                torch.float64 if isinstance(timestep, float) else torch.int64
            )
            timesteps = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif timestep.dim() == 0:
            timesteps = timestep.unsqueeze(0).to(sample.device)
        else:
            timesteps = timestep

        # Broadcast to batch dimension (ONNX/CoreML compatible)
        timesteps = timesteps.expand(sample.shape[0])

        # Sinusoidal projection
        t_emb = self.time_proj(timesteps)
        
        # Cast to sample dtype for mixed precision compatibility
        t_emb = t_emb.to(dtype=sample.dtype)

        # MLP embedding
        return self.time_embedding(t_emb, None)
