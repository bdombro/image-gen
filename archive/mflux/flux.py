#!/usr/bin/env python3
"""
FLUX.2 image generation using mflux (MLX on Apple Silicon)

Status: Abandoned bc the mflux cli seems to do a much better quality job (shrug). If I really need/want python, could compare this to
`mflux/.venv/lib/python3.13/site-packages/mflux/models/flux2/cli/flux2_edit_generate.py`. But for now, doesn't seem worth extra
effort.

Usage:
  uv run flux.py "A cybernetic knight, pixel art, 16-bit"
  uv run flux.py "fantasy landscape" --steps 20 --guidance 4.0 --width 512 --height 512
  uv run flux.py "portrait" --seed 123 -o output.png
"""

import argparse
from pathlib import Path

from mflux.models.common.config import ModelConfig
# from mflux.models.flux2.variants import Flux2Klein
from mflux.models.flux2.variants import Flux2KleinEdit


def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2 image generation using mflux"
    )
    parser.add_argument(
        "prompt", nargs="*",
        help="Text prompt describing the image to generate"
    )
    parser.add_argument(
        "-o", "--output", default="dist/output.png",
        help="Output image path (default: auto-generated from prompt)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--steps", "--num-inference-steps", type=int, default=10, dest="num_inference_steps",
        help="Number of inference steps (default: 10, recommended: 15-20)"
    )
    parser.add_argument(
        "--guidance", type=float, default=3.5,
        help="CFG guidance scale (default: 3.5)"
    )
    parser.add_argument(
        "--width", type=int, default=1024,
        help="Output width in pixels (default: 1024)"
    )
    parser.add_argument(
        "--height", type=int, default=1024,
        help="Output height in pixels (default: 1024)"
    )
    parser.add_argument(
        "-i", "--image-paths", default=None, nargs="+",
        help="Input image path(s) for img2img mode (can specify multiple)"
    )
    
    # NOTE: mflux strength is INVERTED vs diffusers convention.
    #   strength=0.3 => start early  => lots of denoising steps => BIG change
    #   strength=0.8 => start late   => few denoising steps    => small change
    #   Formula: remaining_steps = steps - int(steps * strength)
    parser.add_argument(
        "--strength", type=float, default=0.6,
        help="Img2img strength (0.0–1.0, default: 0.5). Higher = more faithful to input."
    )

    args = parser.parse_args()

    # Join prompt tokens if provided, otherwise use default
    prompt = " ".join(args.prompt).strip() if args.prompt else "A sprite of a alchemist, side profile, pixel art style, 16-bit"

    # Auto-detect width/height from first input image if not explicitly set
    if args.image_paths and (args.width == 1024 or args.height == 1024):
        from PIL import Image
        with Image.open(args.image_paths[0]) as img:
            if args.width == 1024:
                args.width = img.width
            if args.height == 1024:
                args.height = img.height

    print(f"🚀 Loading FLUX.2 Klein")
    model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_9b())
    # model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_4b())

    print(f"✨ Generating…")
    print(f"   Prompt : {prompt}")
    print(f"   Seed   : {args.seed}")
    print(f"   Steps  : {args.num_inference_steps}")
    print(f"   Guidance: {args.guidance}")
    print(f"   Strength: {args.strength if args.image_paths else 'N/A'}")
    print(f"   Size   : {args.width}×{args.height}")
    if args.image_paths:
        print(f"   Inputs : {', '.join(args.image_paths)}  (strength={args.strength})")
    print(f"   Output : {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    image = model.generate_image(
        prompt=prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        image_paths=list(args.image_paths) if args.image_paths else None,
        image_strength=args.strength if args.image_paths else None,
    )

    image.save(args.output)
    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    main()