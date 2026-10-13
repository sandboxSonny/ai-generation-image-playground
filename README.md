# Character Roleplay Chatbot

A Gradio-based roleplay chatbot that combines a local LLM with image generation. Create a fictional character with a consistent appearance, then chat with them — the AI can generate matching scene images on the fly. Everything runs locally.

## Features

- **Character creation** — generate a reference portrait for your character
- **Roleplay chat** — the LLM acts as your character (system prompt + personality)
- **Tool use** — the LLM calls a `generate_image` tool to produce scene images that respect the character's appearance
- **Multiple backends** — NVIDIA/AMD GPUs (SDXL via diffusers) and Apple Silicon (FLUX via MLX)
- **Per-session state** — each browser session has its own character
- **Multi-page Gradio UI** — character creation and chat on separate tabs

## Requirements

- Python 3.10+
- A local OpenAI-compatible LLM server, such as [LM Studio](https://lmstudio.ai/) or `mlx_lm.server`
- One of:
  - **NVIDIA (CUDA) or AMD (ROCm) GPU** — RealVisXL V4.0 (SDXL) needs ~8 GB VRAM minimum, fp16 variant. Developed on ROCm.
  - **Apple Silicon Mac (MLX)** — FLUX.1 [schnell] via [mflux](https://github.com/filipstrand/mflux), using an 8-bit quantized model that needs roughly 12 GB of unified memory on top of whatever the LLM uses. Tested on a 48 GB M-series Mac.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt        # NVIDIA / AMD
pip install -r requirements-mlx.txt    # Apple Silicon (MLX)
```

## Running

### 1. Start a local LLM

The app talks to any OpenAI-compatible endpoint and uses tool calling, so pick a model that supports function calling.

**LM Studio** (default, `http://localhost:1234/v1`): load a model and start the local server. If you enable authentication in LM Studio, put your API token in `.env` as `LLM_API_KEY`. Set `LLM_MODEL` to the identifier shown by `lms ps`.

**mlx-lm** (Apple Silicon):

```bash
mlx_lm.server --model <mlx-community/some-model> --port 8080
LLM_BASE_URL=http://localhost:8080/v1 LLM_MODEL=<mlx-community/some-model> python roleplay_chatbot.py
```

> **Memory tip:** the LLM and the image model share memory on Apple Silicon. LM Studio defaults to a very large context window and several parallel slots, which reserves a lot of memory. For this app a small context is plenty, for example:
> `lms load <model> --context-length 8192 --parallel 1`

### 2. Launch the app

```bash
python roleplay_chatbot.py
```

Opens at `http://127.0.0.1:7860`.

## Configuration

Settings are read from environment variables, or from a `.env` file in the project root (copy `.env.example`; `.env` is gitignored).

| Variable | Default | Meaning |
|---|---|---|
| `IMAGE_BACKEND` | `auto` | `mlx`, `cuda` (CUDA/ROCm) or `mps`. `auto` picks `mlx` on Apple Silicon, else `cuda`. `mps` (SDXL on Apple GPU) is untested |
| `MLX_IMAGE_MODEL` | `schnell` | mflux model name (`dev` is gated; use it with `MLX_MODEL_PATH=` empty) |
| `MLX_MODEL_PATH` | `dhairyashil/FLUX.1-schnell-mflux-8bit` | Hugging Face repo or local path of a pre-quantized model. Set it to empty to use the official `black-forest-labs` repo instead (requires `hf auth login` and accepting its license) |
| `MLX_QUANTIZE` | `8` | Quantization bits, only used when `MLX_MODEL_PATH` is empty |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name sent to the endpoint |
| `LLM_API_KEY` | `lm-studio` | API key sent to the endpoint. If LM Studio has authentication enabled, use your LM Studio API token |
| `GRADIO_HOST` | `127.0.0.1` | Interface to bind. The app has **no authentication**, so only set `0.0.0.0` on a network you trust |
| `GRADIO_PORT` | `7860` | Port to listen on |

The MLX backend uses FLUX rather than RealVisXL, so negative prompts are ignored and results look different from the SDXL backends. FLUX schnell also ignores the guidance value and always runs 4 steps.

## Architecture

```
roleplay_chatbot.py        — Main app: Gradio UI, per-session character state, chat + tool-call logic
backends.py                — Image backends (diffusers CUDA/MPS, mflux MLX) and configuration
scripts/generate_samples.py — Generate a few sample images with the selected backend
scripts/check_gpu.py        — GPU / MLX detection check
references/                — Generated character portraits (gitignored)
images/                    — Output of generate_samples.py (gitignored)
```

## How It Works

1. **Create Character** — Describe appearance + personality → a reference portrait is generated.
2. **Chat** — Messages are sent to the local LLM with the character's personality as the system prompt.
3. **Image Generation** — When the LLM decides to generate an image, it calls the `generate_image` tool with a scene description. The character's appearance is prepended to the prompt so every image stays consistent, and the result is shown next to the chat.

## Models and licences

This project's code is MIT-licensed (see `LICENSE`). The models it downloads have their own terms; check them before any use beyond personal experimentation:

- [RealVisXL V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0) (SDXL backends)
- [FLUX.1 [schnell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (MLX backend). The default downloads a community-quantized copy, [`dhairyashil/FLUX.1-schnell-mflux-8bit`](https://huggingface.co/dhairyashil/FLUX.1-schnell-mflux-8bit), because the official repo is gated. This is unofficial third-party weights; point `MLX_MODEL_PATH` at the official repo if you'd rather not use it.
- Whichever LLM you run locally.

## Limitations

- Local, single-user demo: no authentication, and it binds to localhost by default.
- Characters and chat history live in memory and are lost on restart.
- This is for fictional characters. Don't use it to generate images of real people.
- Image generation takes roughly 10–60 seconds depending on hardware (about 11 s with FLUX schnell on Apple Silicon).
- `requirements.txt` is a full freeze of a ROCm environment (PyTorch + GPU libraries), so it's large and platform-specific.
