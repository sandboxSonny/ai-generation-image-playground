import os
import platform
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()  # read settings (e.g. LLM_API_KEY) from a local .env file

# IMAGE_BACKEND: "auto" | "mlx" | "cuda" | "mps"
IMAGE_BACKEND = os.environ.get("IMAGE_BACKEND", "auto").lower()
MLX_MODEL = os.environ.get("MLX_IMAGE_MODEL", "schnell")
MLX_QUANTIZE = int(os.environ.get("MLX_QUANTIZE", "8"))
# Pre-quantized, ungated schnell mirror (the official black-forest-labs repo is gated).
# Set MLX_MODEL_PATH="" to load MLX_IMAGE_MODEL from its official repo instead (needs HF login).
MLX_MODEL_PATH = os.environ.get("MLX_MODEL_PATH", "dhairyashil/FLUX.1-schnell-mflux-8bit")

# Gradio server. Defaults to localhost only; the app has no authentication,
# so set GRADIO_HOST=0.0.0.0 only on a network you trust.
GRADIO_HOST = os.environ.get("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))

# LLM endpoint: LM Studio (1234) by default; mlx_lm.server uses http://localhost:8080/v1
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")


def resolve_backend():
    if IMAGE_BACKEND != "auto":
        return IMAGE_BACKEND
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "cuda"


class DiffusersBackend:
    """SDXL (RealVisXL) via diffusers on CUDA/ROCm or Apple MPS."""

    def __init__(self, device):
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.torch = torch
        self.device = device
        print(f"Loading RealVisXL on {device}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)
        self.pipe.vae.enable_slicing()

    def generate(self, prompt, negative_prompt, steps, guidance, seed=None):
        generator = None
        if seed is not None:
            generator = self.torch.Generator(self.device).manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]


class MLXBackend:
    """FLUX via mflux (MLX) on Apple Silicon.

    FLUX has no negative prompt, and schnell ignores guidance and needs few
    steps, so those arguments are accepted but not used.
    """

    def __init__(self):
        # MLX streams are thread-local and Gradio calls handlers from worker
        # threads, so load and run the model on one dedicated thread.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._load).result()

    def _load(self):
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1

        if MLX_MODEL_PATH:
            print(f"Loading FLUX from {MLX_MODEL_PATH} on MLX...")
            self.model = Flux1(
                model_path=MLX_MODEL_PATH,
                model_config=ModelConfig.from_name(MLX_MODEL),
            )
        else:
            print(f"Loading FLUX ({MLX_MODEL}, {MLX_QUANTIZE}-bit) on MLX...")
            self.model = Flux1.from_name(MLX_MODEL, quantize=MLX_QUANTIZE)
        self.steps = 4 if MLX_MODEL == "schnell" else 25

    def generate(self, prompt, negative_prompt, steps, guidance, seed=None):
        return self._executor.submit(self._generate, prompt, seed).result()

    def _generate(self, prompt, seed):
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        return self.model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=self.steps,
            height=1024,
            width=1024,
        ).image


def load_image_backend():
    backend = resolve_backend()
    if backend == "mlx":
        return MLXBackend()
    return DiffusersBackend(backend)
