# Character Roleplay Chatbot

A Gradio-based roleplay chatbot that combines an LLM with Stable Diffusion image generation. Create a character with a consistent appearance, then chat with them — the AI can generate matching scene images on the fly.

## Features

- **Character creation** — generate a consistent reference portrait using RealVisXL V4.0
- **Roleplay chat** — the LLM acts as your character (system prompt + personality)
- **Tool use** — the LLM calls `generate_image` to produce scene images that respect the character's appearance
- **Multi-page Gradio UI** — character creation and chat on separate tabs

## Requirements

- **GPU** — NVIDIA GPU with CUDA or AMD GPU with ROCm (tested with ROCm)
  - RealVisXL V4.0 (SDXL-based) needs ~8 GB VRAM minimum
  - fp16 variant recommended
- Python 3.10+

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

### 1. Start a local LLM (optional — uses LM Studio)

[LM Studio](https://lmstudio.ai/) is recommended. Start the local server with a model like GPT-OSP or similar (20B+ params recommended).

Default config points to `http://localhost:1234/v1`.

### 2. Launch the app

```bash
python roleplay_chatbot.py
```

Opens at `http://localhost:7860`.

## Architecture

```
roleplay_chatbot.py  — Main app: Gradio UI, character state, chat logic
test_generate.py     — Standalone image generation test
test_gpu.py          — GPU/ROCm detection check
references/          — Generated character portraits (gitignored)
images/              — Generated scene images (gitignored)
```

## How It Works

1. **Create Character** — Describe appearance + personality → generates a reference portrait via Stable Diffusion XL
2. **Chat** — Messages are sent to the local LLM with the character's personality as the system prompt
3. **Image Generation** — When the LLM decides to generate an image, it provides a scene description. The character's appearance is prepended to the prompt so every image stays consistent.

## Notes

- The `requirements.txt` is large due to PyTorch + full CUDA/ROCm stacks. This is expected for GPU-accelerated ML.
- Image generation takes 10–60 seconds depending on GPU. Be patient.
