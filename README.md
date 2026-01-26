# ComfyUI-IPAdapter-WAN 2.0

This extension adapts the [InstantX IP-Adapter for SD3.5-Large](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter) to work with **Wan 2.1** and other UNet-based video/image models in **ComfyUI**.

Unlike the original SD3 version (which depends on `joint_blocks` from MMDiT), this version performs **sampling-time identity conditioning** by dynamically injecting into **attention layers** — making it compatible with models like **Wan 2.1**, **AnimateDiff**, and other non-SD3 pipelines.

---

## 🚀 Features

### Core Functionality
- 🔁 Injects identity embeddings during sampling via attention block patching
- 🧠 Works with Wan 2.1 and other UNet-style models (no SD3/MMDiT required)
- 🛠️ Built on top of ComfyUI's IPAdapter framework
- 🎨 Enables consistent face/identity across frames in video workflows

### Version 2.0 Upgrades
- ⏱️ **Timestep-based Scheduling**: Fine-grained control over when IP conditioning is active during denoising (start/end percentage)
- 🎯 **Type-Safe Architecture**: Extensive type hints, dataclasses, and runtime validation for better reliability
- 💾 **Enhanced Memory Management**: FP16 by default, explicit device placement, optimized batched processing
- 🔧 **Structured Configuration**: Centralized config system with `IPAdapterConfig` and `VisionEncoderConfig`
- 📊 **Time-Conditioned Resampler**: Advanced `TimeResampler` with Adaptive Layer Normalization (adaLN) for timestep-aware conditioning
- 🛡️ **Fail-Fast Validation**: Input validation at construction time to catch errors early
- 📚 **Comprehensive Documentation**: Detailed docstrings with computational complexity analysis
- 🎨 **True Dynamic Resolution Support**: Perceiver-style resampler architecture enables native support for variable-resolution inputs from SigLIP2 so400m naflex - no fixed sequence length assumptions

---

## 📦 Installation

1. Clone the repo into your ComfyUI custom nodes directory:

```bash
git clone https://github.com/your-username/ComfyUI-IPAdapter-WAN.git
```

2. Download the required model weights:
- Download the IP-Adapter weights:
  
  - [`ip-adapter.bin`](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter/blob/main/ip-adapter.bin)
  
  - Place it in: `ComfyUI/models/ipadapter/`

- Download the CLIP Vision model:
  
  - **SigLIP2 so400m patch16 naflex** (recommended): [`google/siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex)
  
  - Place the model file in: `ComfyUI/models/clip_vision/`
  
  - This model supports dynamic resolution with native flexible (naflex) attention, providing better performance for variable aspect ratios.

*(Note: This node only supports SigLIP2 so400m patch16 naflex. The vision encoder type is automatically set to SigLIP2 - no configuration needed.)*

**Dynamic Resolution**: The IPAdapter architecture fully supports variable-resolution inputs. The Perceiver-style resampler compresses any sequence length from SigLIP2 to a fixed number of queries, enabling true dynamic resolution support without hardcoded size assumptions.

---

## 🧠 How It Works

Wan models use a UNet structure instead of the DiT transformer blocks used in SD3. To make IPAdapter work with Wan:

- The extension scans all attention blocks (modules with `.to_q` and `.to_k`) dynamically.

- It injects IPAdapter's attention processors (`IPAttnProcessor`) directly into those blocks.

- **Version 2.0 Enhancement**: Identity embeddings are updated based on the current sampling timestep using a **time-conditioned resampler** (`TimeResampler`) with:
  - Adaptive Layer Normalization (adaLN) for timestep-aware feature modulation
  - **Perceiver-style cross-attention** for efficient compression of vision embeddings - this enables **true dynamic resolution support** by compressing variable-length sequences to fixed queries
  - Timestep embedding integration for denoising-aware conditioning

- **Dynamic Resolution Support**: The resampler architecture is specifically designed to handle variable-length vision encoder outputs (from SigLIP2 so400m naflex). It compresses any sequence length to a fixed number of queries (64), making it resolution-agnostic. This means:
  - ✅ Works with any input resolution/aspect ratio from SigLIP2
  - ✅ No hardcoded sequence length assumptions
  - ✅ Efficient O(n) scaling with variable token counts

- **Timestep Scheduling**: The extension respects `start_percent` and `end_percent` parameters, only applying IP conditioning during the specified denoising range.

This means it works **without requiring joint_blocks or specific architectural assumptions** — making it plug-and-play for many custom models.

---

## 🛠 Usage

1. In ComfyUI, use the following nodes:
   
   - `Load IPAdapter WAN Model` - Loads the IP-Adapter checkpoint with SigLIP2 so400m configuration
   
   - `Apply IPAdapter WAN Model` - Applies IP conditioning to your diffusion model

2. Connect the `CLIP Vision` embedding (from a face image) and your diffusion model to the adapter.

   - **Note**: This node automatically uses SigLIP2 so400m configuration - no vision encoder selection needed.

3. **New in 2.0**: Configure timestep scheduling:
   - `start_percent` (default: 0.0): When to start applying IP conditioning (0.0 = beginning of denoising)
   - `end_percent` (default: 1.0): When to stop applying IP conditioning (1.0 = end of denoising)
   - This allows fine-grained control over when identity conditioning is active during the denoising process

4. Use a weight of **~0.5** as a good starting point (adjustable from -1.0 to 5.0).

5. You can apply this in video workflows to maintain consistent identity across frames.

---

## 📁 Example Workflows

Example `.json` workflows will be available soon in the `workflows/` folder.

---

## ✅ Compatibility

| Model            | Status              |
| ---------------- | ------------------- |
| Wan 2.1          | ✅ Works             |
| AnimateDiff      | ✅ Works             |
| SD3 / SDXL       | ❌ Use original repo |
| Any UNet variant | ✅ Likely to work    |

---

## ✨ Version 2.0 Technical Improvements

### Architecture Enhancements
- **Type-Safe Design**: Protocol-based interfaces, dataclasses for configuration, comprehensive type hints
- **Memory Optimization**: FP16 by default, efficient batched processing, zero-copy state sharing
- **Error Handling**: Fail-fast validation with clear error messages
- **Modular Configuration**: Centralized config system enabling easy hyperparameter tuning

### Performance Optimizations
- **Efficient Attention**: Uses `F.scaled_dot_product_attention` for hardware-accelerated attention
- **Memory-Efficient Reshaping**: Fused tensor operations to minimize memory copies
- **Lazy Evaluation**: IP conditioning computed only when active (based on timestep schedule)

### Developer Experience
- **Comprehensive Documentation**: Detailed docstrings with computational complexity analysis
- **Extensible Design**: Easy to add new vision encoder types or modify architecture
- **Runtime Validation**: Input validation prevents common errors before execution

---

## 🔧 TODOs

- Allow multiple adapters without conflict

- Auto-detect model parameters (hidden size, num layers)

- Convert `.bin` to `safetensors` format

- Add more workflows for different models

- Support additional vision encoder types beyond SigLIP2

---

## 🧑‍💻 Credits

- Adapted from: [InstantX IPAdapter for SD3.5](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter)

- Version 2.0 enhancements include:
  - Type-safe architecture with comprehensive validation
  - Time-conditioned resampler with adaLN
  - Timestep-based scheduling for fine-grained control
  - Enhanced memory management and performance optimizations

---

## 📝 Changelog

### Version 2.0
- ✨ Added timestep-based scheduling (start_percent/end_percent)
- ✨ Implemented time-conditioned resampler with Adaptive Layer Normalization
- ✨ Enhanced type safety with dataclasses, enums, and protocols
- ✨ Improved memory management (FP16 default, optimized batching)
- ✨ Added comprehensive documentation and error handling
- ✨ Structured configuration system for easier customization

---

Feel free to contribute or suggest improvements via GitHub Issues or Pull Requests.
