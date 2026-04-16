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

import torch
import torch.nn as nn
import folder_paths

from .models.resampler import TimeResampler
from .models.jointblock import JointBlockIPWrapper, IPAttnProcessor


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
    def _unwrap_module(module: Any) -> Any:
        """Unwrap common wrapper attributes (DisTorch2/Distributed/quant wrappers)."""
        seen_ids = set()
        current = module
        while current is not None and id(current) not in seen_ids:
            seen_ids.add(id(current))
            next_module = None
            for attr in ("module", "model", "inner_model", "unet", "wrapped_module"):
                candidate = getattr(current, attr, None)
                if isinstance(candidate, nn.Module):
                    next_module = candidate
                    break
            if next_module is None:
                break
            current = next_module
        return current

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
    model = _unwrap_module(diffusion_model)

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
        print(f"[{NODE_APPLY_NAME}] {message}")
        logger.info("[%s] %s", NODE_APPLY_NAME, message)

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
        raw_timestep = args["timestep"].flatten()[0].detach().float().cpu().item()
        t_percent = 1.0 - (raw_timestep / float(timestep_schedule_max))
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
            
            # Scale timestep to model's expected range
            timestep = args["timestep"] * timestep_schedule_max
            
            # Compute IP embeddings with timestep conditioning
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)

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
            # Emit on active-state entry and at coarse 5% progression buckets.
            progress_bucket = int(t_percent * 20)
            if debug_state["last_active"] is not True or debug_state["last_bucket"] != progress_bucket:
                emit_debug(
                    (
                        "active"
                        f" step={debug_state['step']}"
                        f" t_percent={t_percent:.4f}"
                        f" range=[{start:.3f},{end:.3f}]"
                        f" weight={ip_options_dict['weight']:.3f}"
                        f" mean_ip_norm={ip_norm:.6f}"
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
    
    logger.debug(f"Patched {proc_idx} attention blocks with IP adapters")
    emit_debug(
        (
            f"resolved_diffusion_model={type(model).__name__} "
            f"patched attention blocks={proc_idx}"
            f" timestep_max={timestep_schedule_max}"
            f" weight={weight:.3f}"
            f" range=[{start:.3f},{end:.3f}]"
        )
    )
    if proc_idx == 0:
        emit_debug(
            "warning: no attention blocks were patched. This may indicate an incompatible "
            "loader/wrapper topology (e.g., distributed/quantized wrapper not exposing JointBlock modules)."
        )


# =============================================================================
# Model Container
# =============================================================================

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
        
        self.state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
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
