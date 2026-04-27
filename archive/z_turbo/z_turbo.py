#!/usr/bin/env python3
"""
Z-Image-Turbo inference script
Usage:
  python z_turbo.py "a cat wearing a hat" -o output.png
  python z_turbo.py "with blue hair" -i input.png -o output.png
  python z_turbo.py "with blue hair" -i input.png -n "blue eyes" -o output.png


  .venv/bin/python z_turbo/z_turbo.py -i input.png -o dist/z_turbo.png --cfg=3.5 --strength=.3 --steps=12 \
    -n "black lines, borders, text" \
    "Top-down pixel art sprite. 16 colors. Crisp edges. Solid chromakey green #00FF00 background"

.venv/bin/python z_turbo/z_turbo.py -i dist/z_turbo.png -o dist/z_turbo3.png --cfg=3.5 --strength=.3 --steps=12 \
    -n "black lines, borders, text" \
    "game character, facing right, side profile view"
"""

import argparse
import os

from PIL import Image
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    ZImagePipeline,
    ZImageControlNetPipeline,
    ZImageControlNetInpaintPipeline,
    ZImageControlNetModel,
)
from diffusers.utils import load_image


def get_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        # PyTorch 2.11 supports bf16 on MPS. Use high watermark to allow
        # the full 6B-parameter model (~20GB) + ControlNet (~7GB) on 48GB systems.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.9")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo — text2img / img2img / controlnet"
    )
    parser.add_argument("prompt", nargs="*", help="prompt text (or pipe via heredoc/stdin)")
    parser.add_argument("-o", "--output", help="Output image path (default: auto-generated from prompt)")
    parser.add_argument("-i", "--input", help="Input image for img2img mode")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (e.g. 42). Omit for random."
    )
    

    # Strength impacts how much of the original image is preserved vs transformed, with lower values being more faithful.
    # Turbo mode works best around 0.5–0.7, while higher values (e.g. 0.8–1.0) are more like pure text-to-image generation
    # with a hint of the input. Only applies to img2img pipelines, but it's easier to have one flag than multiple pipeline-specific ones.
    parser.add_argument(
        "-s", "--strength", type=float, default=0.6,
        help="Img2img strength (0.0–1.0, default 0.6)"
    )
    
    # ControlNet parameters — only applies if a control image is provided. Use to guide composition, pose, depth, or other aspects of the 
    # generation with an additional image input. The control image should be a simple representation of the desired structure (e.g. line 
    # drawing, depth map) that the model can follow, while the prompt describes the style and content. The control scale adjusts how 
    # strongly the model adheres to the control image, with 0.0 ignoring it completely and 1.0 following it closely. Turbo mode often works 
    # well around 0.5–0.75 for a balanced blend of creativity and control.
    parser.add_argument("-c", "--control-img", help="Control image for ControlNet")
    parser.add_argument(
        "--control-scale", type=float, default=0.75,
        help="ControlNet conditioning scale (0.0–1.0, default 0.75)"
    )
    
    # Inference steps control the number of denoising iterations during generation. Turbo mode is designed to produce high-quality results 
    # with fewer steps than traditional pipelines, often performing best in the 4–12 range. Fewer steps (e.g. 4–6) can yield faster results 
    # with a more artistic, less detailed style, while more steps (e.g. 10–12) can enhance detail and fidelity at the cost of longer 
    # generation times. The default of 9 is a good starting point for balancing speed and quality.
    parser.add_argument(
        "--steps", type=int, default=9,
        help="Inference steps (Turbo works best at 4–12, default 9)"
    )

    # Guidance scale (CFG) controls how strongly the model follows the text prompt. In Turbo mode, you can set this to 0.0 for pure image-driven 
    # generation without any text guidance, which can yield more creative and unexpected results based solely on the input image and control 
    # image (if provided). For a more traditional text-to-image or img2img experience with adherence to the prompt, values around 2.0–3.5 are 
    # common. The default is None, which means it will be 0.0 for text-to-image (fully turbo) and 2.5 for img2img (balanced).
    parser.add_argument(
        "--cfg", "--guidance-scale", type=float, default=None, dest="guidance_scale",
        help="CFG scale (0.0 = no CFG for pure Turbo mode; 2.0–3.5 for prompt adherence in img2img)"
    )

    parser.add_argument(
        "-n", "--negative-prompt", default=None,
        help="Negative prompt (what to avoid, e.g. 'blue eyes blurry')"
    )

    parser.add_argument(
        "--width", type=int,
        help="Output width (default: input image width, or model default)"
    )

    parser.add_argument(
        "--height", type=int,
        help="Output height (default: input image height, or model default)"
    )

    args = parser.parse_args()
    device, dtype = get_device()

    prompt = args.prompt
    negative_prompt = args.negative_prompt

    # Auto-generate output filename if not provided
    if args.output is None:
        slug = prompt.strip().lower()
        slug = "".join(c if c.isalnum() or c in " -_" else "" for c in slug)
        slug = slug.replace(" ", "-")[:40].strip("-")
        args.output = f"output_{slug or 'generated'}.png"

    print(f"🚀 Loading Z-Image-Turbo on {device}…")
    base = "Tongyi-MAI/Z-Image-Turbo"

    has_input = args.input is not None
    has_ctrl = args.control_img is not None

    # ------------------------------------------------------------------
    # 1. Pick the right pipeline for the job
    # ------------------------------------------------------------------
    if has_ctrl:
        print("🛠️  Loading Z-Image ControlNet model…")
        controlnet = ZImageControlNetModel.from_pretrained(
            "hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
            torch_dtype=dtype,
        )
        if has_input:
            # img2img + ControlNet → use the inpaint pipeline (it supports both)
            pipe = ZImageControlNetInpaintPipeline.from_pretrained(
                base, controlnet=controlnet, torch_dtype=dtype, low_cpu_mem_usage=False,
            )
        else:
            pipe = ZImageControlNetPipeline.from_pretrained(
                base, controlnet=controlnet, torch_dtype=dtype, low_cpu_mem_usage=False,
            )
    elif has_input:
        pipe = AutoPipelineForImage2Image.from_pretrained(base, torch_dtype=dtype, low_cpu_mem_usage=False)
    else:
        pipe = ZImagePipeline.from_pretrained(base, torch_dtype=dtype, low_cpu_mem_usage=False)

    # Memory optimizations
    pipe.to(device)
    pipe.enable_attention_slicing()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if device == "mps":
        torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # 2. Determine output dimensions: default to input size or 512²
    # ------------------------------------------------------------------
    input_img = None
    if has_input:
        print(f"🖼️  Input image  : {args.input}  (strength={args.strength})")
        input_img = load_image(args.input).convert("RGB")
        if args.width is None:
            args.width = input_img.width
        if args.height is None:
            args.height = input_img.height

    # ------------------------------------------------------------------
    # 3. Build call kwargs
    # ------------------------------------------------------------------
    # Z-Image-Turbo is a distilled consistency model that breaks with CFG > 0.
    # Always default to 0.0 (pure Turbo mode). The prompt still influences generation
    # through cross-attention, but CFG scaling causes white/garbage output.
    pipe_kwargs = dict(
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else 0.0,
    )

    if args.seed is not None:
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(args.seed)

    if negative_prompt:
        pipe_kwargs["negative_prompt"] = negative_prompt

    if args.width is not None:
        pipe_kwargs["width"] = args.width
    if args.height is not None:
        pipe_kwargs["height"] = args.height

    if has_input:
        pipe_kwargs["image"] = input_img
        # The ControlNet inpaint pipeline doesn't have a strength param,
        # but ZImageImg2ImgPipeline does — only pass it when it'll be accepted.
        if not has_ctrl:
            pipe_kwargs["strength"] = args.strength

    if has_ctrl:
        print(f"🎮 Control image : {args.control_img}  (scale={args.control_scale})")
        pipe_kwargs["control_image"] = load_image(args.control_img).convert("RGB")
        pipe_kwargs["controlnet_conditioning_scale"] = args.control_scale
        # Inpaint pipeline requires a mask; provide a white mask to transform the whole image
        if has_input:
            pipe_kwargs["mask_image"] = Image.new("L", (args.width or input_img.width, args.height or input_img.height), 255)

    # ------------------------------------------------------------------
    # 3. Generate
    # ------------------------------------------------------------------
    print("✨ Generating…")
    with torch.inference_mode():
        result = pipe(**pipe_kwargs).images[0]

    result.save(args.output)
    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()