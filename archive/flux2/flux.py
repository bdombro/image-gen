#!/usr/bin/env python3
"""
FLUX.2 image generation using standard HuggingFace Diffusers

This model accepts an image param for conditioning (similar to ControlNet's image conditioning,
but built into the main pipeline). We support this as a separate --input-control image to keep
it conceptually distinct from the img2img-style noise blending of the main input image, and
because it can be used with or without an img2img input. It basically uses the image as a style
reference to guide the generation, and can be used for things like consistent character design
across multiple prompts, or applying a specific style to a generation. The control scale parameter
controls how strongly the model follows the control image, with 0.0 being no influence and 1.0 
being full influence.

We can also manipulate the initial latents from an image to make the model start-from an image.
This is supported via the --input image and strength param, which blends the encoded image latents 
with noise to varying degrees.


Usage:
  uv run flux.py "A cybernetic knight, pixel art, 16-bit"
  uv run flux.py "fantasy landscape" --steps 20 --guidance 4.0 --width 512 --height 512
  uv run flux.py "portrait" --seed 123 -o output.png
  uv run flux.py --input input.png "facing right"               # img2img (latent noise blend)
  uv run flux.py --input-control ref.png "same style"            # Fill conditioning
  uv run flux.py --input input.png --input-control ref.png ...   # both at once
"""



import argparse
import os

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_latents


config = {
    "height_default": 1024,
    "width_default": 1024,
    "model_id": "black-forest-labs/FLUX.2-klein-4B",
    "seed_default": 42,
    "steps_default": 15,
    "guidance_default": 3.5,
    "strength_default": 0.6,
}

def _encode_vae_image(pipe, pixel_vals, generator):
    """Encode pixels to latents matching Flux2KleinPipeline._encode_vae_image."""
    image_latent = retrieve_latents(
        pipe.vae.encode(pixel_vals), generator=generator, sample_mode="argmax"
    )
    # Patchify before batch norm (bn operates on 128-channel patched latents)
    image_latent = pipe._patchify_latents(image_latent)
    # Apply batch normalization
    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
        image_latent.device, image_latent.dtype
    )
    bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    )
    image_latent = (image_latent - bn_mean) / bn_std
    return image_latent


def get_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        # MPS can have memory issues with large models; these env vars help
        # mitigate that by adjusting the memory management strategy.

        # High watermark ratio determines when PyTorch starts aggressively caching memory
        # Is a percentage of total GPU memory which pytorch is allowed to use before it
        # starts caching aggressively. 0 = no-cap
        # Url: https://pytorch.org/docs/stable/notes/mps.html#memory-management
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

        # Low watermark ratio determines when PyTorch starts releasing memory
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32


class FluxEngine:
    """Reusable FLUX.2 generation engine. Loads the model once; generate many times."""

    def __init__(self, model_id: str | None = None):
        self.device, self.dtype = get_device()
        self.model_id = model_id or config["model_id"]

        print(f"🚀 Loading {self.model_id} on {self.device}…")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        )
        self.pipe.to(self.device)
        if self.device in ("mps", "cuda"):
            self.pipe.enable_attention_slicing()

    def generate(
        self,
        prompt: str,
        seed: int = config["seed_default"],
        steps: int = config["steps_default"],
        guidance: float = config["guidance_default"],
        width: int | None = None,
        height: int | None = None,
        input_image: "str | PIL.Image.Image | None" = None,
        control_image: "str | PIL.Image.Image | None" = None,
        strength: float = config["strength_default"],
    ) -> "PIL.Image.Image":
        """Run a single generation and return the result PIL image.

        Args:
            prompt: Text prompt.
            seed: Random seed.
            steps: Number of denoising steps.
            guidance: CFG guidance scale.
            width: Output width. If None, inferred from input images or config default.
            height: Output height. If None, inferred from input images or config default.
            input_image: File path or PIL Image for img2img (latent noise blend).
            control_image: File path or PIL Image for Fill/ControlNet-style conditioning.
            strength: Img2img strength — lower = more faithful to input.

        Returns:
            PIL.Image.Image — the generated image (not saved).
        """
        # Normalize image params: accept str (path) or PIL Image
        _input_img = None
        _control_img = None
        if isinstance(input_image, str):
            _input_img = load_image(input_image).convert("RGB")
            print(f"   Input  : {input_image}  (strength={strength})")
        elif input_image is not None:
            _input_img = input_image
            print(f"   Input  : <PIL Image>  (strength={strength})")
        if isinstance(control_image, str):
            _control_img = load_image(control_image).convert("RGB")
            print(f"   Control: {control_image}")
        elif control_image is not None:
            _control_img = control_image
            print(f"   Control: <PIL Image>")

        # Resolve dimensions
        if width is None or height is None:
            _img = _input_img or _control_img
            if _img is not None:
                if width is None:
                    width = _img.width
                if height is None:
                    height = _img.height
            width = width or config["width_default"]
            height = height or config["height_default"]

        print(f"✨ Generating…")
        print(f"   Prompt : {prompt}")
        print(f"   Seed   : {seed}")
        print(f"   Steps  : {steps}")
        print(f"   Size   : {width}×{height}")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs: dict = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "width": width,
            "height": height,
        }

        with torch.inference_mode():
            if _control_img is not None:
                kwargs["image"] = _control_img

            if _input_img is not None:
                pixel_vals = torch.from_numpy(
                    np.array(_input_img, dtype=np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype)
                pixel_vals = torch.nn.functional.interpolate(
                    pixel_vals,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

                image_latent = _encode_vae_image(self.pipe, pixel_vals, generator)
                noise = torch.randn_like(image_latent, generator=generator)

                t = 1.0 - strength
                kwargs["latents"] = (1.0 - t) * noise + t * image_latent

            image = self.pipe(**kwargs).images[0]

        return image


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2 image generation using standard Diffusers"
    )
    parser.add_argument(
        "prompt", nargs="*",
        help="Text prompt describing the image to generate"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image path (default: auto-generated from prompt)"
    )
    parser.add_argument(
        "--seed", type=int, default=config["seed_default"],
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--steps", "--num-inference-steps", type=int, default=config["steps_default"], dest="num_inference_steps",
        help="Number of inference steps (default: 15, recommended: 15-20)"
    )
    parser.add_argument(
        "--guidance", type=float, default=config["guidance_default"],
        help="CFG guidance scale 0-3.5 (default: 3.5)"
    )
    parser.add_argument(
        "--width", type=int,
        help=f"Output width in pixels (default: {config['width_default']})"
    )
    parser.add_argument(
        "--height", type=int,
        help=f"Output height in pixels (default: {config['height_default']})"
    )
    parser.add_argument(
        "-i", "--input", default=None,
        help="Input image path for img2img mode (blends with noise based on --strength)"
    )
    parser.add_argument(
        "--input-control", default=None,
        help="Input image path for Fill/ControlNet-style conditioning (image= parameter)"
    )
    parser.add_argument(
        "--strength", type=float, default=config["strength_default"],
        help="Img2img strength (0.0–1.0, default: 0.6). Lower = more faithful to input."
    )
    parser.add_argument(
        "--quantize", type=int, default=4, choices=[4, 8],
        help="Quantization arg (Ignored for diffusers, kept for CLI compatibility)"
    )

    args = parser.parse_args()

    # Join prompt tokens if provided, otherwise use default
    prompt = " ".join(args.prompt).strip() if args.prompt else "A sprite of an alchemist, side profile, pixel art style, 16-bit"

    # Auto-generate output filename if not provided
    if args.output is None:
        slug = prompt.strip().lower()
        slug = "".join(c if c.isalnum() or c in " -_" else "" for c in slug)
        slug = slug.replace(" ", "-")[:40].strip("-")
        args.output = f"{slug or 'generated'}.png"

    engine = FluxEngine()

    # If no explicit width/height, load images early to infer dimensions
    width = args.width
    height = args.height
    if width is None or height is None:
        _img_path = args.input or args.input_control
        if _img_path:
            _img = load_image(_img_path)
            width = width or _img.width
            height = height or _img.height
    width = width or config["width_default"]
    height = height or config["height_default"]

    result = engine.generate(
        prompt=prompt,
        seed=args.seed,
        steps=args.num_inference_steps,
        guidance=args.guidance,
        width=width,
        height=height,
        input_image=args.input,
        control_image=args.input_control,
        strength=args.strength,
    )

    result.save(args.output)
    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
