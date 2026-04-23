"""
IPAdapter WAN Integration Module for ComfyUI

Architectural Philosophy:
- Decorator pattern for non-invasive model patching
- Lazy evaluation of IP conditioning based on timestep scheduling
- Memory-efficient embedding management with batched processing

Cognitive Profile of Original Implementation:
- Pragmatic approach prioritizing functionality over abstraction
- Domain expertise in diffusion model conditioning pipelines
- Procedural mindset with some OOP encapsulation

Enhanced Design:
- Type-safe interfaces with runtime validation
- Configurable vision encoder support (CLIP, SigLIP2)
- Improved memory management for multi-GPU scenarios
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
from enum import Enum, auto
import math
import itertools
import types
import pickle

import torch
import torch.nn as nn
import folder_paths

from .models.resampler import TimeResampler
from .models.jointblock import JointBlockIPWrapper, IPAttnProcessor

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None


# =============================================================================
# Configuration and Constants
# =============================================================================

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)

logger = logging.getLogger(__name__)
NODE_APPLY_NAME = "Apply IPAdapter WAN Model"
PATCH_SESSION_COUNTER = itertools.count(1)


class VisionEncoderType(Enum):
    """Supported vision encoder architectures."""
    SIGLIP2_SO400M = auto()  # SigLIP2 patch16 so400m naflex (dynamic resolution, variable tokens)
    CLIP_VIT_H = auto()  # OpenCLIP ViT-H/14 style fixed-token encoder


@dataclass(frozen=True)
class VisionEncoderConfig:
    """
    Configuration for vision encoder compatibility.
    
    SigLIP2 so400m naflex advantages:
    - Dynamic input resolution (shape-optimized)
    - Better fine-grained visual understanding
    - Improved efficiency for variable aspect ratios
    
    Computational Note:
    SigLIP2 naflex uses NaFlex (Native Flexible) attention which adapts
    to input resolution without fixed positional embeddings, enabling
    O(n) scaling with image tokens vs O(n²) for fixed-resolution models.
    """
    embedding_dim: int
    supports_dynamic_resolution: bool = False
    max_sequence_length: Optional[int] = None
    
    @classmethod
    def from_encoder_type(cls, encoder_type: VisionEncoderType) -> "VisionEncoderConfig":
        configs = {
            VisionEncoderType.SIGLIP2_SO400M: cls(
                embedding_dim=1152,  # so400m variant
                supports_dynamic_resolution=True,
                max_sequence_length=None,  # naflex: variable
            ),
            VisionEncoderType.CLIP_VIT_H: cls(
                embedding_dim=1024,  # OpenCLIP ViT-H hidden size
                supports_dynamic_resolution=False,
                max_sequence_length=257,
            ),
        }
        return configs[encoder_type]


@dataclass
class IPAdapterConfig:
    """
    Unified configuration for IPAdapter instantiation.
    
    Design Rationale:
    Centralizing configuration enables:
    1. Validation at construction time (fail-fast)
    2. Serialization for checkpointing
    3. Easy A/B testing of hyperparameters
    """
    # Resampler architecture
    resampler_dim: int = 1280
    resampler_depth: int = 4
    resampler_dim_head: int = 64
    resampler_heads: int = 20
    num_queries: int = 64
    output_dim: int = 2432
    ff_mult: int = 4
    
    # Timestep embedding
    timestep_in_dim: int = 320
    timestep_flip_sin_to_cos: bool = True
    timestep_freq_shift: int = 0
    
    # Attention processor
    hidden_size: int = 2432
    cross_attention_dim: int = 2432
    head_dim: int = 64
    timesteps_emb_dim: int = 1280
    
    # Vision encoder (default to SigLIP2 for modern workflows)
    vision_encoder: VisionEncoderType = VisionEncoderType.SIGLIP2_SO400M
    embedding_dim_override: Optional[int] = None
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from vision encoder config."""
        if self.embedding_dim_override is not None:
            return self.embedding_dim_override
        return VisionEncoderConfig.from_encoder_type(self.vision_encoder).embedding_dim


@dataclass
class IPOptions:
    """
    Runtime options for IP conditioning injection.
    
    Mutable state container passed to attention wrappers.
    Using dataclass over dict for type safety and IDE support.
    """
    hidden_states: Optional[torch.Tensor] = None
    t_emb: Optional[torch.Tensor] = None
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility with existing wrappers."""
        return {
            "hidden_states": self.hidden_states,
            "t_emb": self.t_emb,
            "weight": self.weight,
        }
    
    def clear(self) -> None:
        """Reset state between timesteps."""
        self.hidden_states = None
        self.t_emb = None


# =============================================================================
# Core Patching Infrastructure
# =============================================================================

class AttentionBlockProtocol(Protocol):
    """Protocol for attention blocks that can be wrapped."""
    def to_q(self, x: torch.Tensor) -> torch.Tensor: ...
    def to_k(self, x: torch.Tensor) -> torch.Tensor: ...


def create_timestep_scheduler(
    start: float,
    end: float,
) -> Callable[[float], bool]:
    """
    Factory for timestep-based conditioning schedulers.
    
    Returns a predicate that determines if IP conditioning should be active.
    
    Computational Complexity: O(1) per evaluation
    
    Args:
        start: Start percentage (0.0 = beginning of denoising)
        end: End percentage (1.0 = end of denoising)
    
    Returns:
        Predicate function: (t_percent) -> bool
    
    Note:
        t_percent is computed as 1 - timestep/max_timestep, so:
        - t_percent = 0.0 at the start of denoising (high noise)
        - t_percent = 1.0 at the end of denoising (low noise)
    """
    # Validate bounds at creation time (fail-fast)
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError(f"start and end must be in [0, 1], got start={start}, end={end}")
    if start > end:
        raise ValueError(f"start must be <= end, got start={start}, end={end}")
    
    def is_active(t_percent: float) -> bool:
        return start <= t_percent <= end
    
    return is_active


def patch_model(
    patcher,
    ip_procs: nn.ModuleList,
    resampler: TimeResampler,
    clip_embeds: torch.Tensor,
    weight: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
) -> None:
    """
    Inject IPAdapter processors into diffusion model attention blocks.
    
    Architecture:
    1. Wraps the UNet forward pass to compute IP embeddings per timestep
    2. Patches individual attention blocks with IP-augmented wrappers
    
    Computational Complexity:
    - Per-timestep overhead: O(num_queries * seq_len * dim) for resampler
    - Per-block overhead: O(seq_len * ip_seq * dim) for IP attention
    
    Memory Overhead:
    - O(batch * num_queries * output_dim) for IP hidden states
    - O(batch * timestep_emb_dim) for timestep embeddings
    
    Args:
        patcher: ComfyUI model patcher instance
        ip_procs: ModuleList of IPAttnProcessor instances
        resampler: TimeResampler for embedding compression
        clip_embeds: Vision embeddings of shape (2, seq_len, embed_dim)
                     Index 0: conditional, Index 1: unconditional (zeros)
        weight: Conditioning strength multiplier
        start: Start percentage for timestep scheduling
        end: End percentage for timestep scheduling
    """
    patch_session_id = next(PATCH_SESSION_COUNTER)

    def _module_stats(module: nn.Module) -> Dict[str, int]:
        jointblock_like = 0
        to_qk_like = 0
        total = 0
        for _, mod in module.named_modules():
            total += 1
            if hasattr(mod, "context_block") and hasattr(mod, "x_block"):
                jointblock_like += 1
            if hasattr(mod, "to_q") and hasattr(mod, "to_k"):
                to_qk_like += 1
        return {
            "total": total,
            "jointblock_like": jointblock_like,
            "to_qk_like": to_qk_like,
        }

    def _resolve_best_module(module: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """Resolve best patching target across wrapper chain (DisTorch2/Distributed/GGUF)."""
        chain: List[Tuple[str, nn.Module]] = []
        seen_ids = set()
        current = module
        while isinstance(current, nn.Module) and id(current) not in seen_ids:
            seen_ids.add(id(current))
            chain.append((type(current).__name__, current))
            next_module = None
            for attr in ("module", "model", "inner_model", "unet", "wrapped_module"):
                candidate = getattr(current, attr, None)
                if isinstance(candidate, nn.Module):
                    next_module = candidate
                    break
            if next_module is None:
                break
            current = next_module

        # Prefer module level with the most JointBlock-like modules.
        best_idx = 0
        best_score: Tuple[int, int, int] = (-1, -1, -1)
        chain_debug: List[Dict[str, Any]] = []
        for idx, (name, mod) in enumerate(chain):
            stats = _module_stats(mod)
            chain_debug.append(
                {
                    "idx": idx,
                    "name": name,
                    "id": id(mod),
                    "total": stats["total"],
                    "jointblock_like": stats["jointblock_like"],
                    "to_qk_like": stats["to_qk_like"],
                }
            )
            score = (stats["jointblock_like"], stats["to_qk_like"], stats["total"])
            if score > best_score:
                best_score = score
                best_idx = idx
        return (chain[best_idx][1] if chain else module), chain_debug

    # Resolve diffusion model robustly across different loaders/wrappers.
    base_model = getattr(patcher, "model", None)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None and hasattr(base_model, "model"):
        diffusion_model = getattr(base_model.model, "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError(
            f"{NODE_APPLY_NAME}: could not resolve diffusion_model from patcher "
            f"(patcher_type={type(patcher).__name__}, base_model_type={type(base_model).__name__})."
        )
    model, unwrap_chain_debug = _resolve_best_module(diffusion_model)

    # Resolve timestep schedule robustly; some wrappers do not expose model_config directly.
    sampling_settings = getattr(getattr(base_model, "model_config", None), "sampling_settings", None)
    if sampling_settings is None and hasattr(base_model, "model"):
        sampling_settings = getattr(getattr(base_model.model, "model_config", None), "sampling_settings", None)
    timestep_schedule_max = 1000
    if isinstance(sampling_settings, dict):
        timestep_schedule_max = int(sampling_settings.get("timesteps", 1000))
    timestep_schedule_max = max(1, timestep_schedule_max)
    
    # Create scheduler predicate
    is_active = create_timestep_scheduler(start, end)
    
    # Mutable state container for cross-block communication
    # NOTE: This must stay as a dict reference shared with all wrappers.
    # Using IPOptions.to_dict() creates a snapshot and breaks updates.
    ip_options_dict: Dict[str, Any] = {
        "hidden_states": None,
        "t_emb": None,
        "weight": weight,
    }
    
    def emit_debug(message: str) -> None:
        # Console visibility is important for ComfyUI users diagnosing workflows.
        print(f"[{NODE_APPLY_NAME}][session={patch_session_id}] {message}")

    emit_debug(
        "patch setup "
        f"patcher_type={type(patcher).__name__} patcher_id={id(patcher)} "
        f"base_model_type={type(base_model).__name__} base_model_id={id(base_model)} "
        f"diffusion_model_type={type(diffusion_model).__name__} diffusion_model_id={id(diffusion_model)}"
    )
    for item in unwrap_chain_debug:
        emit_debug(
            "unwrap_chain "
            f"idx={item['idx']} name={item['name']} id={item['id']} "
            f"total={item['total']} jointblock_like={item['jointblock_like']} "
            f"to_qk_like={item['to_qk_like']}"
        )

    debug_state = {
        "last_active": None,
        "last_bucket": None,
        "step": 0,
    }

    def ddit_wrapper(
        forward: Callable,
        args: Dict[str, Any],
    ) -> torch.Tensor:
        """
        UNet forward wrapper that computes IP embeddings.
        
        This wrapper intercepts the diffusion model forward pass to:
        1. Check if IP conditioning should be active for current timestep
        2. Compute IP embeddings via the resampler if active
        3. Store embeddings in ip_options for attention block access
        
        Design Note:
        Using a closure over ip_options enables zero-copy state sharing
        with attention wrappers, avoiding the overhead of passing through
        the entire call stack.
        """
        # Compute timestep percentage (1 - t/max gives progress through denoising)
        timestep_tensor = args["timestep"].detach().float()
        raw_timestep = timestep_tensor.flatten()[0].cpu().item()
        max_abs_timestep = timestep_tensor.abs().max().cpu().item()
        sigma_like_timesteps = max_abs_timestep <= 1.5

        # Normalize once and reuse for both scheduling and resampler units.
        if sigma_like_timesteps:
            timestep_normalized = raw_timestep
            timestep_for_resampler = args["timestep"] * timestep_schedule_max
        else:
            timestep_normalized = raw_timestep / float(timestep_schedule_max)
            timestep_for_resampler = args["timestep"]

        t_percent = 1.0 - timestep_normalized
        t_percent = max(0.0, min(1.0, t_percent))
        debug_state["step"] += 1
        
        if is_active(t_percent):
            # Compute batch size accounting for classifier-free guidance
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            cond_or_uncond = torch.as_tensor(
                args["cond_or_uncond"], device=clip_embeds.device
            )
            
            # Select embeddings based on CFG indices and expand to batch
            # cond_or_uncond: [0] for conditional, [1] for unconditional, [0, 1] for both
            embeds = clip_embeds[cond_or_uncond]
            embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
            
            # Compute IP embeddings with timestep conditioning
            image_emb, t_emb = resampler(embeds, timestep_for_resampler, need_temb=True)

            # Prevent unconditional branch leakage:
            # the resampler can produce non-zero outputs from zero embeds due to biases.
            # If unconditional branch carries IP signal, CFG can cancel most of the effect,
            # making the weight slider appear unresponsive.
            uncond_mask = torch.repeat_interleave(cond_or_uncond == 1, batch_size, dim=0)
            image_emb = image_emb.clone()
            image_emb[uncond_mask] = 0.0
            
            ip_options_dict["hidden_states"] = image_emb
            ip_options_dict["t_emb"] = t_emb
            ip_norm = image_emb.float().norm(dim=-1).mean().item()
            ip_mean = image_emb.float().mean().item()
            ip_std = image_emb.float().std().item()
            uncond_ratio = uncond_mask.float().mean().item()
            # Emit on active-state entry and at coarse 5% progression buckets.
            progress_bucket = int(t_percent * 20)
            if (
                debug_state["last_active"] is not True
                or debug_state["last_bucket"] != progress_bucket
                or debug_state["step"] <= 3
            ):
                emit_debug(
                    (
                        "active"
                        f" step={debug_state['step']}"
                        f" t_percent={t_percent:.4f}"
                        f" range=[{start:.3f},{end:.3f}]"
                        f" weight={ip_options_dict['weight']:.3f}"
                        f" mean_ip_norm={ip_norm:.6f}"
                        f" mean={ip_mean:.6f}"
                        f" std={ip_std:.6f}"
                        f" uncond_ratio={uncond_ratio:.3f}"
                        f" cond_or_uncond={list(map(int, cond_or_uncond.cpu().tolist()))}"
                        f" sigma_like={sigma_like_timesteps}"
                        f" input_shape={tuple(args['input'].shape)}"
                    )
                )
            debug_state["last_active"] = True
            debug_state["last_bucket"] = progress_bucket
        else:
            ip_options_dict["hidden_states"] = None
            ip_options_dict["t_emb"] = None
            if debug_state["last_active"] is not False:
                emit_debug(
                    (
                        "inactive"
                        f" step={debug_state['step']}"
                        f" t_percent={t_percent:.4f}"
                        f" outside range=[{start:.3f},{end:.3f}]"
                    )
                )
            debug_state["last_active"] = False
            debug_state["last_bucket"] = None
        
        return forward(args["input"], args["timestep"], **args["c"])
    
    patcher.set_model_unet_function_wrapper(ddit_wrapper)
    
    # Patch attention blocks with IP wrappers
    # Strategy: Round-robin assignment of processors to blocks
    proc_idx = 0
    for name, module in model.named_modules():
        # WAN uses MMDiT JointBlocks. Patch only modules compatible with JointBlockIPWrapper.
        is_jointblock_like = hasattr(module, "context_block") and hasattr(module, "x_block")
        if is_jointblock_like:
            wrapper = JointBlockIPWrapper(
                module,
                ip_procs[proc_idx % len(ip_procs)],
                ip_options_dict,
            )
            patcher.set_model_patch_replace(wrapper, name)
            proc_idx += 1
            if proc_idx <= 60:
                emit_debug(
                    f"patched module idx={proc_idx} name='{name}' class='{type(module).__name__}' "
                    f"module_id={id(module)} proc_idx={(proc_idx - 1) % len(ip_procs)}"
                )
    
    logger.debug(f"Patched {proc_idx} attention blocks with IP adapters")
    model_stats = _module_stats(model) if isinstance(model, nn.Module) else {
        "total": 0,
        "jointblock_like": 0,
        "to_qk_like": 0,
    }
    emit_debug(
        (
            f"resolved_diffusion_model={type(model).__name__} "
            f"module_stats(total={model_stats['total']}, "
            f"jointblock_like={model_stats['jointblock_like']}, "
            f"to_qk_like={model_stats['to_qk_like']}) "
            f"patched attention blocks={proc_idx}"
            f" timestep_max={timestep_schedule_max}"
            f" weight={weight:.3f}"
            f" range=[{start:.3f},{end:.3f}]"
        )
    )
    if proc_idx == 0:
        # Extra diagnostics for incompatible wrappers/topologies.
        debug_lines: List[str] = []
        class_counts: Dict[str, int] = {}
        for module_name, module_obj in model.named_modules():
            cls_name = type(module_obj).__name__
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            if "block" in module_name.lower() or "joint" in cls_name.lower():
                attrs = []
                for attr in ("context_block", "x_block", "to_q", "to_k", "attn", "attn2"):
                    if hasattr(module_obj, attr):
                        attrs.append(attr)
                if attrs:
                    debug_lines.append(
                        f"candidate module='{module_name}' class='{cls_name}' attrs={attrs}"
                    )
            if len(debug_lines) >= 50:
                break
        # Show the top module classes to identify architectural patterns.
        top_classes = sorted(
            class_counts.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:25]
        emit_debug(f"top_module_classes={top_classes}")

        # Broader attention-like scan (q/k naming variants used by different loaders/models).
        attention_attr_pairs = [
            ("to_q", "to_k"),
            ("q_proj", "k_proj"),
            ("query", "key"),
            ("wq", "wk"),
            ("q", "k"),
        ]
        attention_candidates: List[str] = []
        for module_name, module_obj in model.named_modules():
            matched = []
            for q_attr, k_attr in attention_attr_pairs:
                if hasattr(module_obj, q_attr) and hasattr(module_obj, k_attr):
                    matched.append((q_attr, k_attr))
            if matched:
                attention_candidates.append(
                    f"attention_candidate module='{module_name}' class='{type(module_obj).__name__}' matches={matched}"
                )
            if len(attention_candidates) >= 80:
                break
        for line in attention_candidates:
            emit_debug(line)

        # WAN fallback: direct monkey-patch of WanSelfAttention modules.
        # This path is used when Comfy patch_replace cannot find JointBlock hooks.
        wan_self_attn_modules: List[Tuple[str, nn.Module]] = [
            (n, m)
            for n, m in model.named_modules()
            if type(m).__name__ == "WanSelfAttention" and hasattr(m, "q") and hasattr(m, "k")
        ]
        if wan_self_attn_modules:
            emit_debug(
                f"fallback: found {len(wan_self_attn_modules)} WanSelfAttention modules; "
                "installing direct forward wrappers"
            )
            for idx, (module_name, attn_module) in enumerate(wan_self_attn_modules):
                if not getattr(attn_module, "_ipadapter_wan_patched", False):
                    attn_module._ipadapter_wan_original_forward = attn_module.forward
                original_forward = attn_module._ipadapter_wan_original_forward
                adapter = ip_procs[idx % len(ip_procs)]

                def patched_forward(self_attn, *f_args, __orig=original_forward, __adapter=adapter, **f_kwargs):
                    out = __orig(*f_args, **f_kwargs)
                    ip_hidden = ip_options_dict.get("hidden_states")
                    t_emb = ip_options_dict.get("t_emb")
                    if ip_hidden is None or t_emb is None:
                        return out

                    if not f_args:
                        return out
                    hidden_states = f_args[0]
                    if not torch.is_tensor(hidden_states) or not torch.is_tensor(out):
                        return out
                    if hidden_states.ndim != 3 or out.ndim != 3:
                        return out

                    # q/k/v projections expected on WAN attention modules.
                    if not (hasattr(self_attn, "q") and hasattr(self_attn, "k") and hasattr(self_attn, "v")):
                        return out
                    try:
                        img_query = self_attn.q(hidden_states)
                        img_key = self_attn.k(hidden_states)
                        img_value = self_attn.v(hidden_states)
                    except Exception:
                        return out

                    if img_query.ndim != 3 or img_key.ndim != 3 or img_value.ndim != 3:
                        return out

                    # Align q/k/v width with adapter hidden size (e.g. WAN self-attn may use 2560 while
                    # IPAdapter uses 2432). We pad/truncate on the feature axis for compatibility.
                    target_dim = int(__adapter.to_k_ip.weight.shape[0])
                    def _align_last_dim(t: torch.Tensor, dim: int) -> torch.Tensor:
                        current = t.shape[-1]
                        if current == dim:
                            return t
                        if current > dim:
                            return t[..., :dim]
                        pad = dim - current
                        return torch.nn.functional.pad(t, (0, pad))

                    img_query = _align_last_dim(img_query, target_dim)
                    img_key = _align_last_dim(img_key, target_dim)
                    img_value = _align_last_dim(img_value, target_dim)

                    head_dim = max(1, int(getattr(__adapter, "head_dim", 64)))
                    if target_dim % head_dim != 0:
                        return out
                    n_heads = max(1, target_dim // head_dim)

                    head_dim_val = max(1, img_value.shape[-1] // n_heads)
                    img_value = img_value.view(img_value.shape[0], img_value.shape[1], n_heads, head_dim_val)

                    ip_delta = __adapter(
                        ip_hidden,
                        img_query,
                        img_key,
                        img_value,
                        t_emb,
                        int(n_heads),
                    )
                    if ip_delta is None:
                        return out

                    # Align output space for WAN attention variants where attention out-width
                    # differs from module forward output width.
                    if ip_delta.shape != out.shape and ip_delta.ndim == 3 and out.ndim == 3:
                        if hasattr(self_attn, "o"):
                            try:
                                projected = self_attn.o(ip_delta)
                                if torch.is_tensor(projected):
                                    ip_delta = projected
                            except Exception:
                                pass
                        if ip_delta.shape != out.shape and ip_delta.shape[0] == out.shape[0] and ip_delta.shape[1] == out.shape[1]:
                            target_dim = out.shape[-1]
                            if ip_delta.shape[-1] > target_dim:
                                ip_delta = ip_delta[..., :target_dim]
                            else:
                                ip_delta = torch.nn.functional.pad(ip_delta, (0, target_dim - ip_delta.shape[-1]))

                    if ip_delta.shape != out.shape:
                        if debug_state["step"] <= 3:
                            emit_debug(
                                f"fallback skip shape_mismatch module='{type(self_attn).__name__}' "
                                f"ip_delta_shape={tuple(ip_delta.shape)} out_shape={tuple(out.shape)}"
                            )
                        return out

                    if debug_state["step"] <= 3:
                        emit_debug(
                            f"fallback apply module='{type(self_attn).__name__}' "
                            f"delta_norm={ip_delta.float().norm().item():.6f} out_norm={out.float().norm().item():.6f}"
                        )
                    return out + ip_delta.to(out.dtype) * float(ip_options_dict.get("weight", 1.0))

                attn_module.forward = types.MethodType(patched_forward, attn_module)
                attn_module._ipadapter_wan_patched = True
                proc_idx += 1
                if proc_idx <= 60:
                    emit_debug(
                        f"fallback patched WanSelfAttention idx={proc_idx} "
                        f"name='{module_name}' class='{type(attn_module).__name__}' module_id={id(attn_module)}"
                    )

        for line in debug_lines:
            emit_debug(line)
        if proc_idx == 0:
            emit_debug(
                "warning: no attention blocks were patched. This may indicate an incompatible "
                "loader/wrapper topology (e.g., distributed/quantized wrapper not exposing JointBlock modules)."
            )


# =============================================================================
# Model Container
# =============================================================================

def _load_ipadapter_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load IPAdapter checkpoint with safe defaults and format-aware fallbacks.

    Priority:
    1) `.safetensors` -> safetensors loader (no pickle execution)
    2) torch.load(weights_only=True)
    3) optional fallback torch.load(weights_only=False) for trusted files
    """
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == ".safetensors":
        if safetensors_load_file is None:
            raise RuntimeError(
                "safetensors checkpoint detected but safetensors is unavailable in this environment."
            )
        return safetensors_load_file(checkpoint_path, device=str(device))

    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except pickle.UnpicklingError as exc:
        logger.warning(
            "weights_only=True load failed for %s (%s). Retrying with weights_only=False. "
            "Use only with trusted files.",
            checkpoint_path,
            exc,
        )
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )


def _normalize_ipadapter_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize checkpoint layouts to:
      {
        "image_proj": {...},
        "ip_adapter": {...},
      }

    Supported raw layouts:
    - nested dict (already normalized)
    - flat safetensors-style keys: "image_proj.*", "ip_adapter.*"
    - wrapped: {"state_dict": {...}}
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    # Already normalized
    if (
        "image_proj" in state_dict
        and "ip_adapter" in state_dict
        and isinstance(state_dict["image_proj"], dict)
        and isinstance(state_dict["ip_adapter"], dict)
    ):
        return state_dict

    image_proj: Dict[str, Any] = {}
    ip_adapter: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("image_proj."):
            image_proj[key[len("image_proj."):]] = value
        elif key.startswith("ip_adapter."):
            ip_adapter[key[len("ip_adapter."):]] = value

    if image_proj and ip_adapter:
        return {
            "image_proj": image_proj,
            "ip_adapter": ip_adapter,
        }

    return state_dict

class WANIPAdapter:
    """
    Container for IPAdapter WAN model components.
    
    Lifecycle:
    1. Load checkpoint from disk
    2. Initialize resampler with architecture config
    3. Initialize attention processors based on checkpoint structure
    4. Load weights into both components
    
    Memory Management:
    - Models are loaded in fp16 by default for memory efficiency
    - Supports explicit device placement for multi-GPU setups
    
    Extensibility:
    - Config-driven architecture enables easy adaptation to new checkpoints
    - Vision encoder type can be specified for SigLIP2 compatibility
    """
    
    def __init__(
        self,
        checkpoint: str,
        device: Union[str, torch.device],
        config: Optional[IPAdapterConfig] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize IPAdapter from checkpoint.
        
        Args:
            checkpoint: Filename of checkpoint in MODELS_DIR
            device: Target device for model placement
            config: Optional architecture config (uses defaults if None)
            dtype: Model dtype (default fp16 for memory efficiency)
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype
        
        # Load checkpoint
        checkpoint_path = os.path.join(MODELS_DIR, checkpoint)
        logger.info(f"Loading IPAdapter checkpoint from {checkpoint_path}")
        
        raw_state_dict = _load_ipadapter_checkpoint(checkpoint_path, self.device)
        self.state_dict = _normalize_ipadapter_state_dict(raw_state_dict)
        if not isinstance(self.state_dict, dict):
            raise RuntimeError(
                f"Unsupported checkpoint format for {checkpoint}. Expected dict-like state_dict."
            )
        if "image_proj" not in self.state_dict or "ip_adapter" not in self.state_dict:
            available_keys = list(self.state_dict.keys())[:20]
            raise RuntimeError(
                "Checkpoint is not WAN-compatible for this node. "
                "Expected WAN-style keys 'image_proj' and 'ip_adapter' (nested or flat-prefixed). "
                f"Found keys (sample): {available_keys}"
            )
        logger.info(
            "Checkpoint key layout normalized: image_proj_keys=%s ip_adapter_keys=%s",
            len(self.state_dict["image_proj"]),
            len(self.state_dict["ip_adapter"]),
        )

        # Auto-infer architecture from checkpoint when explicit config isn't provided.
        if config is None:
            try:
                self.config = self._infer_config_from_checkpoint(self.state_dict)
                logger.info(
                    "Inferred IPAdapter config from checkpoint: embedding_dim=%s, "
                    "resampler_dim=%s, depth=%s, num_queries=%s, output_dim=%s, "
                    "head_dim=%s, timesteps_emb_dim=%s",
                    self.config.embedding_dim,
                    self.config.resampler_dim,
                    self.config.resampler_depth,
                    self.config.num_queries,
                    self.config.output_dim,
                    self.config.head_dim,
                    self.config.timesteps_emb_dim,
                )
            except Exception as exc:
                logger.warning(
                    "Could not infer checkpoint architecture automatically (%s). "
                    "Falling back to default SigLIP2 config.",
                    exc,
                )
                self.config = IPAdapterConfig(vision_encoder=VisionEncoderType.SIGLIP2_SO400M)
        else:
            self.config = config
        
        # Initialize and load resampler
        self.resampler = self._build_resampler()
        missing_keys, unexpected_keys = self.resampler.load_state_dict(
            self.state_dict["image_proj"],
            strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys when loading resampler: {missing_keys}")
        if unexpected_keys:
            logger.debug(f"Unexpected keys in resampler checkpoint (ignored): {unexpected_keys}")
        
        # Initialize and load attention processors
        self.procs = self._build_processors()
        # Load with strict=False to handle any extra keys in checkpoint
        missing_keys, unexpected_keys = self.procs.load_state_dict(
            self.state_dict["ip_adapter"], 
            strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys when loading IP adapter: {missing_keys}")
        if unexpected_keys:
            logger.debug(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")
        
        logger.info(
            f"Loaded IPAdapter with {len(self.procs)} processors, "
            f"resampler queries={self.config.num_queries}"
        )

    @staticmethod
    def _infer_config_from_checkpoint(state_dict: Dict[str, Any]) -> IPAdapterConfig:
        """
        Infer architecture parameters directly from checkpoint tensors.

        This enables compatibility with multiple WAN IPAdapter variants
        (e.g. SigLIP2-based and ViT-H/Plus-style checkpoints) without
        requiring hard-coded per-model config tables.
        """
        image_proj = state_dict["image_proj"]
        ip_adapter = state_dict["ip_adapter"]

        # Resampler core dimensions
        latents = image_proj["latents"]
        resampler_dim = latents.shape[-1]
        num_queries = latents.shape[1]
        embedding_dim = image_proj["proj_in.weight"].shape[1]
        output_dim = image_proj["proj_out.weight"].shape[0]

        # Depth from transformer layers
        layer_indices = {
            int(key.split(".")[1])
            for key in image_proj.keys()
            if key.startswith("layers.")
        }
        resampler_depth = (max(layer_indices) + 1) if layer_indices else 0

        # FF multiplier from first layer feed-forward matrix
        ff_inner = image_proj["layers.0.1.1.weight"].shape[0]
        ff_mult = max(1, ff_inner // resampler_dim)

        # Attention head dimensions from IP processor weights
        first_proc_prefix = min(int(k.split(".")[0]) for k in ip_adapter.keys())
        first_proc = str(first_proc_prefix)
        hidden_size = ip_adapter[f"{first_proc}.to_k_ip.weight"].shape[0]
        timesteps_emb_dim = ip_adapter[f"{first_proc}.norm_ip.linear.weight"].shape[1]
        head_dim = ip_adapter[f"{first_proc}.norm_q.weight"].shape[0]
        # Infer Perceiver attention inner dimensions from resampler weights, not UNet hidden size.
        # to_q: (inner_dim, resampler_dim), to_kv: (2 * inner_dim, resampler_dim)
        attn_inner_dim = image_proj["layers.0.0.to_q.weight"].shape[0]
        if image_proj.get("layers.0.0.to_kv.weight", None) is not None:
            kv_inner_dim = image_proj["layers.0.0.to_kv.weight"].shape[0] // 2
            if kv_inner_dim != attn_inner_dim:
                logger.warning(
                    "Checkpoint attention inner-dim mismatch (to_q=%s vs to_kv=%s). "
                    "Using to_q-derived value.",
                    attn_inner_dim,
                    kv_inner_dim,
                )

        if attn_inner_dim % head_dim == 0:
            resampler_dim_head = head_dim
            resampler_heads = max(1, attn_inner_dim // head_dim)
        else:
            # Fallback for unusual checkpoints: choose a valid factorization of inner_dim.
            inferred_dim_head = math.gcd(attn_inner_dim, resampler_dim)
            resampler_dim_head = max(1, inferred_dim_head)
            resampler_heads = max(1, attn_inner_dim // resampler_dim_head)

        # Best-effort encoder type label from embedding width
        if embedding_dim == 1024:
            encoder_type = VisionEncoderType.CLIP_VIT_H
        else:
            encoder_type = VisionEncoderType.SIGLIP2_SO400M

        return IPAdapterConfig(
            resampler_dim=resampler_dim,
            resampler_depth=resampler_depth,
            resampler_dim_head=resampler_dim_head,
            resampler_heads=resampler_heads,
            num_queries=num_queries,
            output_dim=output_dim,
            ff_mult=ff_mult,
            hidden_size=hidden_size,
            cross_attention_dim=hidden_size,
            head_dim=head_dim,
            timesteps_emb_dim=timesteps_emb_dim,
            vision_encoder=encoder_type,
            embedding_dim_override=embedding_dim,
        )
    
    def _build_resampler(self) -> TimeResampler:
        """
        Construct TimeResampler from config.
        
        Architecture Notes:
        - Perceiver-style cross-attention compresses variable-length input
        - Timestep conditioning via adaLN enables denoising-aware features
        - Output dimension matches attention hidden size for direct injection
        """
        resampler = TimeResampler(
            dim=self.config.resampler_dim,
            depth=self.config.resampler_depth,
            dim_head=self.config.resampler_dim_head,
            heads=self.config.resampler_heads,
            num_queries=self.config.num_queries,
            embedding_dim=self.config.embedding_dim,
            output_dim=self.config.output_dim,
            ff_mult=self.config.ff_mult,
            timestep_in_dim=self.config.timestep_in_dim,
            timestep_flip_sin_to_cos=self.config.timestep_flip_sin_to_cos,
            timestep_freq_shift=self.config.timestep_freq_shift,
        )
        resampler.eval()
        resampler.to(self.device, dtype=self.dtype)
        return resampler
    
    def _build_processors(self) -> nn.ModuleList:
        """
        Construct attention processors based on checkpoint structure.
        
        Discovery Logic:
        Checkpoint keys are formatted as "{block_idx}.{param_name}"
        We count unique block indices to determine processor count.
        """
        # Discover number of processors from checkpoint keys
        block_indices = set(
            key.split(".")[0] 
            for key in self.state_dict["ip_adapter"].keys()
        )
        n_procs = len(block_indices)
        
        procs = nn.ModuleList([
            IPAttnProcessor(
                hidden_size=self.config.hidden_size,
                cross_attention_dim=self.config.cross_attention_dim,
                ip_hidden_states_dim=self.config.output_dim,
                ip_encoder_hidden_states_dim=self.config.output_dim,
                head_dim=self.config.head_dim,
                timesteps_emb_dim=self.config.timesteps_emb_dim,
            ).to(self.device, dtype=self.dtype)
            for _ in range(n_procs)
        ])
        
        return procs


# =============================================================================
# ComfyUI Node Definitions
# =============================================================================

class IPAdapterWANLoader:
    """
    ComfyUI node for loading IPAdapter WAN models.
    
    Supports multiple vision encoder types for forward compatibility
    with SigLIP2 and other architectures.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter"),),
                "provider": (["cuda", "cpu", "mps"],),
            },
        }
    
    RETURN_TYPES = ("IP_ADAPTER_WAN_INSTANTX",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"
    
    def load_model(
        self,
        ipadapter: str,
        provider: str,
    ) -> Tuple[WANIPAdapter]:
        """
        Load IPAdapter model and auto-detect checkpoint architecture.
        
        Args:
            ipadapter: Checkpoint filename
            provider: Device provider (cuda/cpu/mps)
        
        Returns:
            Tuple containing loaded WANIPAdapter instance
        """
        logger.info(
            "Loading InstantX IPAdapter WAN model: %s (auto-detect architecture)",
            ipadapter,
        )
        model = WANIPAdapter(ipadapter, provider, config=None)
        
        return (model,)


class ApplyIPAdapterWAN:
    """
    ComfyUI node for applying IPAdapter conditioning to diffusion models.
    
    Supports timestep-based scheduling for fine-grained control over
    when IP conditioning is active during the denoising process, with
    runtime validation that vision embedding dimensions match the loaded
    IPAdapter checkpoint.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IP_ADAPTER_WAN_INSTANTX",),
                "image_embed": ("CLIP_VISION_OUTPUT",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "InstantXNodes"
    
    def apply_ipadapter(
        self,
        model,
        ipadapter: WANIPAdapter,
        image_embed,
        weight: float,
        start_percent: float,
        end_percent: float,
    ) -> Tuple[Any]:
        """
        Apply IPAdapter conditioning to model.
        
        Embedding Preparation:
        - Extract penultimate hidden states from vision encoder output
        - Concatenate with zero tensor for classifier-free guidance
        - Move to adapter device with appropriate dtype
        
        Args:
            model: ComfyUI model wrapper
            ipadapter: Loaded WANIPAdapter instance
            image_embed: Vision encoder output (CLIP_VISION_OUTPUT)
            weight: Conditioning strength
            start_percent: Start of active timestep range
            end_percent: End of active timestep range
        
        Returns:
            Tuple containing patched model
        """
        # Clone model to avoid mutating original
        new_model = model.clone()
        
        # Extract embeddings (penultimate layer for richer features)
        image_embed_tensor = image_embed.penultimate_hidden_states
        
        # Validate vision embedding compatibility with loaded checkpoint.
        embed_dim = image_embed_tensor.shape[-1]
        expected_embed_dim = ipadapter.config.embedding_dim
        if embed_dim != expected_embed_dim:
            raise ValueError(
                f"{NODE_APPLY_NAME}: image_embed dim mismatch. "
                f"Loaded IPAdapter expects embed_dim={expected_embed_dim}, "
                f"but received embed_dim={embed_dim}. "
                f"Please use a matching CLIP vision encoder/checkpoint pair."
            )
        
        # Prepare CFG-compatible embeddings: [conditional, unconditional]
        # Unconditional uses zeros to enable classifier-free guidance
        embeds = torch.cat(
            [image_embed_tensor, torch.zeros_like(image_embed_tensor)],
            dim=0,
        ).to(ipadapter.device, dtype=ipadapter.dtype)
        
        # Apply patching
        print(
            f"[{NODE_APPLY_NAME}] setup device={ipadapter.device} dtype={ipadapter.dtype} "
            f"weight={weight:.3f} range=[{start_percent:.3f},{end_percent:.3f}] "
            f"embed_shape={tuple(image_embed_tensor.shape)} expected_embed_dim={expected_embed_dim}"
        )
        patch_model(
            new_model,
            ipadapter.procs,
            ipadapter.resampler,
            embeds,
            weight=weight,
            start=start_percent,
            end=end_percent,
        )
        
        return (new_model,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "IPAdapterWANLoader": IPAdapterWANLoader,
    "ApplyIPAdapterWAN": ApplyIPAdapterWAN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterWANLoader": "Load IPAdapter WAN Model",
    "ApplyIPAdapterWAN": "Apply IPAdapter WAN Model",
}
