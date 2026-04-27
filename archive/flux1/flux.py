#!/usr/bin/env python3
"""
FLUX.1 image generation with ControlNet + Redux using HuggingFace Diffusers.

Four pipeline modes, auto-selected based on args:
  text2img  — FluxPipeline (no input images)
  img2img   — FluxImg2ImgPipeline (--input only)
  control   — FluxControlNetImg2ImgPipeline (--input + --input-control)
  redux     — FluxPriorReduxPipeline encodes ref image → FluxPipeline

Redux mode (--ref-image) uses BFL's official FLUX.1-Redux-dev to encode a
reference image into prompt embeddings. This preserves character appearance
(colors, proportions, style) while allowing full pose freedom through the
text prompt.

Usage:
  uv run flux.py "A cybernetic knight, pixel art, 16-bit"
  uv run flux.py --ref-image ref.png "top-down view"   # Redux: character ref + pose prompt
  uv run flux.py -i sprite.png --input-control weapon.png "holding rifle"
"""

import argparse
import os

import cv2
import numpy as np
import torch
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxControlNetImg2ImgPipeline,
    FluxControlNetModel,
    FluxPriorReduxPipeline,
)
from diffusers.utils import load_image
from PIL import Image


config = {
    "height_default": 1024,
    "width_default": 1024,
    "model_id": "black-forest-labs/FLUX.1-dev",
    "controlnet_id": "InstantX/FLUX.1-dev-controlnet-canny",
    "redux_id": "black-forest-labs/FLUX.1-Redux-dev",
    "seed_default": 42,
    "steps_default": 28,
    "guidance_default": 3.5,
    "strength_default": 0.6,
    "controlnet_scale_default": 0.6,
    "canny_low": 50,
    "canny_high": 150,
    "redux_scale": 0.7,
}


def get_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32


def compute_canny(image: Image.Image, low: int | None = None, high: int | None = None) -> Image.Image:
    """Convert a PIL image to a canny edge map for ControlNet conditioning."""
    low = low if low is not None else config["canny_low"]
    high = high if high is not None else config["canny_high"]
    gray = np.array(image.convert("L"))
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(edges).convert("RGB")


class FluxEngine:
    """Reusable FLUX.1 generation engine. Loads pipelines lazily; generate many times.

    Auto-routes based on ref_image, input_image, and control_image:

    =========== ============= =============== =================================
    ref_image   input_image   control_image   Pipeline
    =========== ============= =============== =================================
    Yes/No      None          None            FluxPipeline (± Redux prior)
    Yes/No      given         None            FluxImg2ImgPipeline (± Redux)
    Yes/No      given         given           FluxControlNetImg2ImgPipeline
    =========== ============= =============== =================================

    When ``ref_image`` is provided, it is encoded via ``FluxPriorReduxPipeline``
    and the resulting embeddings are fed to the pipeline instead of the raw text
    prompt — guiding character appearance while the text prompt controls pose.
    """

    def __init__(
        self,
        model_id: str | None = None,
        controlnet_id: str | None = None,
        redux_id: str | None = None,
    ):
        self.device, self.dtype = get_device()
        self.model_id = model_id or config["model_id"]
        self.controlnet_id = controlnet_id or config["controlnet_id"]
        self.redux_id = redux_id or config["redux_id"]
        self._pipes: dict[tuple, object] = {}
        self._controlnet: FluxControlNetModel | None = None
        self._redux_pipe: FluxPriorReduxPipeline | None = None

    # ── lazy pipeline loading ─────────────────────────────────────────────

    def _get_pipe(self, pipe_cls: type, **extra_kwargs) -> object:
        """Load and cache a pipeline."""
        key = (pipe_cls, frozenset(extra_kwargs.items()))
        if key not in self._pipes:
            print(f"🚀 Loading {pipe_cls.__name__}…")
            if pipe_cls is FluxControlNetImg2ImgPipeline:
                if self._controlnet is None:
                    print(f"   Loading ControlNet: {self.controlnet_id}")
                    self._controlnet = FluxControlNetModel.from_pretrained(
                        self.controlnet_id, torch_dtype=self.dtype,
                    )
                pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    controlnet=self._controlnet,
                    torch_dtype=self.dtype,
                    **extra_kwargs,
                )
            elif pipe_cls is FluxPriorReduxPipeline:
                pipe = FluxPriorReduxPipeline.from_pretrained(
                    self.redux_id,
                    torch_dtype=self.dtype,
                    **extra_kwargs,
                )
            else:
                pipe = pipe_cls.from_pretrained(
                    self.model_id, torch_dtype=self.dtype, **extra_kwargs,
                )
            if self.device in ("mps", "cuda"):
                pipe.enable_attention_slicing()
                if hasattr(pipe, "vae"):
                    pipe.vae.enable_tiling()
            pipe.to(self.device)
            self._pipes[key] = pipe
        return self._pipes[key]

    def _get_redux_embeds(
        self,
        ref_image: Image.Image,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a reference image + prompt via Redux prior.

        Returns (prompt_embeds, pooled_prompt_embeds) for any Flux pipeline."""
        pipe = self._get_pipe(FluxPriorReduxPipeline)
        out = pipe(image=ref_image, prompt=prompt)
        return out.prompt_embeds, out.pooled_prompt_embeds

    # ── image normalisation ───────────────────────────────────────────────

    @staticmethod
    def _resolve_image(
        image: str | Image.Image | None,
        label: str,
    ) -> Image.Image | None:
        if isinstance(image, str):
            result = load_image(image).convert("RGB")
            print(f"   {label}: {image}")
            return result
        if image is not None:
            print(f"   {label}: <PIL Image>")
            return image
        return None

    @staticmethod
    def _resolve_dims(
        width: int | None,
        height: int | None,
        *images: Image.Image | None,
    ) -> tuple[int, int]:
        for img in images:
            if img is not None:
                width = width or img.width
                height = height or img.height
                if width and height:
                    break
        return width or config["width_default"], height or config["height_default"]

    # ── main generation ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        seed: int = config["seed_default"],
        steps: int = config["steps_default"],
        guidance: float = config["guidance_default"],
        width: int | None = None,
        height: int | None = None,
        input_image: str | Image.Image | None = None,
        control_image: str | Image.Image | None = None,
        strength: float = config["strength_default"],
        controlnet_scale: float | None = None,
        ref_image: str | Image.Image | None = None,
    ) -> Image.Image:
        """Run a single generation.

        Args:
            prompt: Text prompt (describes pose/scene when ref_image is used).
            ref_image: Reference image — Redux prior encodes character
                appearance without constraining pose.

        Returns:
            PIL.Image.Image — the generated image (not saved).
        """
        _input_img = self._resolve_image(input_image, "Input")
        _control_raw = self._resolve_image(control_image, "Control")
        _ref_img = self._resolve_image(ref_image, "Ref Image")
        width, height = self._resolve_dims(width, height, _input_img, _control_raw, _ref_img)
        ctrl_scale = controlnet_scale if controlnet_scale is not None else config["controlnet_scale_default"]

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # ── Redux encoding ────────────────────────────────────────────────
        if _ref_img is not None:
            print(f"🔮 Encoding ref image via Redux prior…")
            prompt_embeds, pooled_prompt_embeds = self._get_redux_embeds(_ref_img, prompt)
            prompt_arg: dict = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
            }
        else:
            prompt_arg = {"prompt": prompt}

        print(f"✨ Generating…")
        print(f"   Prompt : {prompt}")
        print(f"   Seed   : {seed}")
        print(f"   Steps  : {steps}")
        print(f"   Size   : {width}×{height}")

        common = {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "width": width,
            "height": height,
        }

        with torch.inference_mode():
            if _input_img is None and _control_raw is None:
                pipe = self._get_pipe(FluxPipeline)
                image = pipe(**common, **prompt_arg).images[0]

            elif _input_img is not None and _control_raw is None:
                pipe = self._get_pipe(FluxImg2ImgPipeline)
                image = pipe(image=_input_img, strength=strength, **common, **prompt_arg).images[0]

            elif _input_img is None and _control_raw is not None:
                pipe = self._get_pipe(FluxImg2ImgPipeline)
                image = pipe(image=_control_raw, strength=strength, **common, **prompt_arg).images[0]

            else:
                canny_img = compute_canny(_input_img)
                pipe = self._get_pipe(FluxControlNetImg2ImgPipeline)
                image = pipe(
                    image=_input_img, control_image=canny_img,
                    controlnet_conditioning_scale=ctrl_scale,
                    strength=strength, **common, **prompt_arg,
                ).images[0]

        return image


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.1 image generation with ControlNet + Redux using Diffusers"
    )
    parser.add_argument("prompt", nargs="*", help="Text prompt describing the image")
    parser.add_argument("-o", "--output", default=None, help="Output image path")
    parser.add_argument("--seed", type=int, default=config["seed_default"])
    parser.add_argument("--steps", "--num-inference-steps", type=int, default=config["steps_default"], dest="num_inference_steps")
    parser.add_argument("--guidance", type=float, default=config["guidance_default"])
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("-i", "--input", default=None, help="Input image for img2img")
    parser.add_argument("--input-control", default=None, help="Triggers ControlNet (canny of --input)")
    parser.add_argument("--ref-image", default=None, help="Reference image — Redux prior encodes character appearance")
    parser.add_argument("--strength", type=float, default=config["strength_default"])
    parser.add_argument("--controlnet-scale", type=float, default=config["controlnet_scale_default"])

    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() or "A pixel art character, SNES style, 16 colors, green screen background"

    if args.output is None:
        safe = prompt.lower().replace(" ", "_")[:30]
        args.output = f"dist/{safe}.png"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    engine = FluxEngine()
    img = engine.generate(
        prompt=prompt, seed=args.seed, steps=args.num_inference_steps,
        guidance=args.guidance, width=args.width, height=args.height,
        input_image=args.input, control_image=args.input_control,
        strength=args.strength, controlnet_scale=args.controlnet_scale,
        ref_image=args.ref_image,
    )
    img.save(args.output)
    print(f"✅ Saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
