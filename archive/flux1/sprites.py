#!/usr/bin/env python3
"""
Sprite generation pipeline using FluxEngine.

Subcommands:
    weapon      Generate a standalone weapon sprite
    character   Generate unarmed character sprite frames
    composite   Composite a weapon onto existing character sprites (alias)
    character-with-weapon
                Composite a weapon onto existing character sprites
    all         Run the full pipeline: weapon → character → composite

Usage:
    uv run python3 sprites.py weapon --prompt "laser rifle" -o weapon.png
    uv run python3 sprites.py character --char-prompt "robot" --out-dir sprites
    uv run python3 sprites.py character-with-weapon --weapon weapon.png --sprite-dir sprites
    uv run python3 sprites.py all --char-prompt "robot" --weapon-prompt "laser rifle"

Python API (zero local deps except flux.py):
    from sprites import run_sprite_gen, run_single_weapon, run_composite_weapon
    run_sprite_gen(char_prompt="knight")
    run_single_weapon("laser rifle", "weapon.png")
    run_composite_weapon("weapon.png", "sprites/", "composited/")
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.dirname(__file__))

from flux import FluxEngine
from PIL import Image

__all__ = [
    "CONFIG",
    "DIRS",
    "FRAMES",
    "build_frames",
    "run_single_weapon",
    "run_sprite_gen",
    "run_composite_weapon",
    "build_parser",
    "main",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG: dict[str, int] = {
    "width": 768,
    "height": 768,
    "steps": 28,
    "seed_base": 100,
}

DIRS = ["front", "right", "up", "down"]

BASE = "SNES pixel art, 16 colors, crisp edges, retro style. Solid chromakey green #00FF00 background."
BASE_WEAPON = "A character holding {WPN} with both hands, left hand gripping the back of the launcher near the stock, right hand gripping the front near the barrel;"

class FrameDef(TypedDict):
    normal: str
    weapon: str


DEFAULT_WPN = "a two-handed sci-fi bazooka"

FRAMES: dict[str, FrameDef] = {
    "idle_front": {
        "normal": f"{BASE} A character standing still, breathing in, hands at sides, front view, directly facing the viewer, face centered, arms relaxed, empty hands, no weapons.",
        "weapon": f"{BASE_WEAPON} standing still, big tube launcher held diagonally across the chest, front view, directly facing the viewer, face centered.",
    },
    "idle_down": {
        "normal": f"{BASE} Top-down overhead view of a character, looking down from above, mostly just the top of the head and shoulders visible, character seen from above, hands empty, holding nothing.",
        "weapon": f"{BASE_WEAPON} looking down from above, mostly just the top of the head and shoulders visible, character seen from above.",
    },
    "walk_1": {
        "normal": f"{BASE} A character stepping forward with right leg, arms swinging naturally, side profile facing the right side of the screen, hands empty, no weapons.",
        "weapon": f"{BASE_WEAPON} stepping forward with right leg, launcher aimed forward at waist level, side profile facing the right side of the screen.",
    },
    "walk_2": {
        "normal": f"{BASE} A character in passing pose, feet together, arms at sides naturally, side profile facing the right side of the screen, hands empty, holding nothing.",
        "weapon": f"{BASE_WEAPON} passing pose, feet together, launcher aimed forward at waist level, side profile facing the right side of the screen.",
    },
    "jump": {
        "normal": f"{BASE} A character at apex of jump, legs raised and tucked, arms raised slightly for balance, side profile facing the right side of the screen, hands open, no weapons.",
        "weapon": f"{BASE_WEAPON} at apex of jump, legs raised and tucked, launcher aimed forward, side profile facing the right side of the screen.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Library functions — Weapon
# ═══════════════════════════════════════════════════════════════════════════════


def build_frames(weapon_desc: str | None = None) -> dict[str, str]:
    """Build weapon compositing prompts with the given weapon description."""
    wpn = weapon_desc or DEFAULT_WPN
    return {name: frame["weapon"].format(WPN=wpn) for name, frame in FRAMES.items()}


def run_single_weapon(
    prompt: str,
    output: str,
    *,
    seed: int | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    engine: FluxEngine | None = None,
) -> str:
    """Generate a single weapon sprite.

    Args:
        prompt: Description of the weapon.
        output: Path to save the PNG.
        seed: Random seed (default: CONFIG["seed_base"]).
        steps: Denoising steps (default: CONFIG["steps"]).
        width: Output width (default: CONFIG["width"]).
        height: Output height (default: CONFIG["height"]).
        engine: Reusable FluxEngine instance (created fresh if omitted).

    Returns:
        Absolute path to the saved PNG.
    """
    seed = seed or CONFIG["seed_base"]
    steps = steps or CONFIG["steps"]
    width = width or CONFIG["width"]
    height = height or CONFIG["height"]

    if engine is None:
        engine = FluxEngine()

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    img = engine.generate(
        prompt=(
            "SNES pixel art, 16 colors, crisp edges, retro style. "
            "Solid chromakey green #00FF00 background. "
            f"A {prompt}, isolated on chromakey green background, "
            "no character, centered, facing right."
        ),
        seed=seed,
        steps=steps,
        width=width,
        height=height,
    )
    img.save(output)
    print(f"✅ Weapon saved to {os.path.abspath(output)}")
    return os.path.abspath(output)


# ═══════════════════════════════════════════════════════════════════════════════
#  Library functions — Character sprites
# ═══════════════════════════════════════════════════════════════════════════════


def run_sprite_gen(
    char_image: str | None = None,
    out_dir: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    seed_base: int | None = None,
    char_prompt: str | None = None,
    engine: FluxEngine | None = None,
) -> list[str]:
    """Generate all sprite sheet frames.

    Args:
        char_image: Path to a reference image for conditioning.
        out_dir: Directory to write output PNGs into. Default is ``dist/character``
            relative to this file.
        width: Output width in pixels (default: CONFIG["width"]).
        height: Output height in pixels (default: CONFIG["height"]).
        steps: Number of denoising steps per frame (default: CONFIG["steps"]).
        seed_base: Base seed; each frame uses seed_base + index
            (default: CONFIG["seed_base"]).
        char_prompt: Text prompt for the character appearance, used instead of
            ``char_image`` if no image is provided.
        engine: Reusable FluxEngine instance (created fresh if omitted).

    Returns:
        List of absolute paths to the generated PNG files.
    """
    if not char_image and not char_prompt:
        raise ValueError("Either char_image or char_prompt must be provided")

    width = width or CONFIG["width"]
    height = height or CONFIG["height"]
    steps = steps or CONFIG["steps"]
    seed_base = seed_base or CONFIG["seed_base"]

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "character")
    os.makedirs(out_dir, exist_ok=True)

    if engine is None:
        print(f"🚀 Initializing FluxEngine (model loaded once for {len(FRAMES)} frames)…")
        engine = FluxEngine()

    # Redux ref image: use the original char_image (or frame-0 hub) for all frames.
    # The original reference gives the strongest visual consistency.
    ref_image: str | Image.Image | None = char_image
    paths: list[str] = []

    for idx, (name, frame) in enumerate(FRAMES.items()):
        path = os.path.join(out_dir, f"{name}.png")
        print(f"\n── Frame {idx + 1}/{len(FRAMES)}: {name} ──")

        img = engine.generate(
            prompt=frame["normal"],
            seed=seed_base + idx,
            steps=steps,
            width=width,
            height=height,
            ref_image=ref_image,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
        paths.append(path)
        print(f"✅ Saved {path}")

        if idx == 0:
            hub_image = img

    print(f"\n✅ All {len(FRAMES)} frames generated in {out_dir}/")

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#  Library functions — Composite weapon onto sprites
# ═══════════════════════════════════════════════════════════════════════════════


def run_composite_weapon(
    weapon_image: str,
    sprite_dir: str,
    out_dir: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    seed_base: int | None = None,
    strength: float = 0.4,
    weapon_prompt: str | None = None,
    engine: FluxEngine | None = None,
) -> list[str]:
    """Composite a weapon onto existing character sprites.

    For each sprite in ``sprite_dir``, generates a new version that has the
    weapon composited via img2img + control conditioning.

    Args:
        weapon_image: Path to the weapon image for Fill/ControlNet conditioning.
        sprite_dir: Directory containing ``frame_*.png`` character sprites.
        out_dir: Output directory for weaponized frames. Default is ``dist/weapon``
            relative to this file.
        width: Output width (default: sprite dimensions).
        height: Output height (default: sprite dimensions).
        steps: Number of denoising steps per frame.
        seed_base: Base seed; each frame uses seed_base + index
            (default: CONFIG["seed_base"]).
        strength: Img2img strength (default: 0.4). Higher = more weapon influence,
            lower = preserves character better.
        engine: Reusable FluxEngine instance (created fresh if omitted).

    Returns:
        List of absolute paths to the generated PNG files.
    """
    width = width or CONFIG["width"]
    height = height or CONFIG["height"]
    steps = steps or CONFIG["steps"]
    seed_base = seed_base or CONFIG["seed_base"]

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "weapon")
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔫 Loading weapon image: {weapon_image}")

    if engine is None:
        print(f"🚀 Initializing FluxEngine (model loaded once)…")
        engine = FluxEngine()

    frames = build_frames(weapon_prompt)
    paths: list[str] = []

    for idx, (name, prompt) in enumerate(frames.items()):
        sprite_path = os.path.join(sprite_dir, f"{name}.png")
        out_path = os.path.join(out_dir, f"{name}.png")
        print(f"\n── Frame {idx + 1}/{len(FRAMES)}: {name} ──")

        img = engine.generate(
            prompt=prompt,
            seed=seed_base + idx,
            steps=steps,
            width=width,
            height=height,
            input_image=sprite_path,
            control_image=weapon_image,
            ref_image=weapon_image,
            strength=strength,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
        paths.append(out_path)
        print(f"✅ Saved {out_path}")

    print(f"\n✅ All {len(frames)} weaponized frames in {out_dir}/")

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    root = argparse.ArgumentParser(
        description="Sprite generation pipeline using FluxEngine",
    )
    root.add_argument(
        "--width", type=int, default=CONFIG["width"],
        help=f"Output width in pixels (default: {CONFIG['width']})",
    )
    root.add_argument(
        "--height", type=int, default=CONFIG["height"],
        help=f"Output height in pixels (default: {CONFIG['height']})",
    )
    root.add_argument(
        "--steps", type=int, default=CONFIG["steps"],
        help=f"Denoising steps (default: {CONFIG['steps']})",
    )
    root.add_argument(
        "--seed-base", type=int, default=CONFIG["seed_base"],
        help=f"Base seed; each frame uses seed_base + index (default: {CONFIG['seed_base']})",
    )

    sub = root.add_subparsers(dest="command", required=True)

    # ── weapon ──
    p_weapon = sub.add_parser("weapon", help="Generate a standalone weapon sprite")
    p_weapon.add_argument(
        "--prompt", default=(
            "futuristic two-handed sci-fi blaster, rectangular body, "
            "barrel on top, short and compact, front-loading muzzle, pixel art weapon"
        ),
        help="Description of the weapon",
    )
    p_weapon.add_argument(
        "-o", "--output", default="dist/weapon.png",
        help="Output path (default: dist/weapon.png)",
    )

    # ── character ──
    p_char = sub.add_parser("character", help="Generate unarmed character sprite frames")
    p_char.add_argument("--char-image", help="Reference image for character conditioning")
    p_char.add_argument(
        "--char-prompt",
        help="Text prompt for character appearance (used if --char-image not given)",
    )
    p_char.add_argument(
        "--out-dir", default=None,
        help="Output directory for frames (default: dist/character)",
    )

    # ── composite / character-with-weapon ──
    p_comp = sub.add_parser(
        "character-with-weapon",
        aliases=["composite"],
        help="Composite a weapon onto existing character sprites"
    )
    p_comp.add_argument(
        "--weapon", required=True,
        help="Path to the weapon image for conditioning",
    )
    p_comp.add_argument(
        "--sprite-dir", required=True,
        help="Directory containing frame_*.png character sprites",
    )
    p_comp.add_argument(
        "--out-dir", default=None,
        help="Output directory for weaponized frames (default: dist/weapon)",
    )
    p_comp.add_argument(
        "--strength", type=float, default=0.4,
        help="Img2img strength (default: 0.4)",
    )
    p_comp.add_argument(
        "--weapon-prompt", default=None,
        help="Weapon description to use in prompts",
    )

    # ── all ──
    p_all = sub.add_parser(
        "all", help="Run the full pipeline: weapon → character → composite"
    )
    p_all.add_argument("--char-image", help="Reference image for character conditioning")
    p_all.add_argument(
        "--char-prompt",
        help="Text prompt for character appearance (used if --char-image not given)",
    )
    p_all.add_argument(
        "--weapon-prompt", default=(
            "futuristic two-handed sci-fi blaster, rectangular body, "
            "barrel on top, short and compact, front-loading muzzle, pixel art weapon"
        ),
        help="Prompt for the weapon sprite",
    )
    p_all.add_argument(
        "--weapon",
        help="Path to an existing weapon sprite (skips weapon generation)",
    )

    return root


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    kwargs = {
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed_base": args.seed_base,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.command == "weapon":
        run_single_weapon(
            prompt=args.prompt,
            output=args.output,
            seed=args.seed_base,
            steps=args.steps,
            width=args.width,
            height=args.height,
        )

    elif args.command == "character":
        if not args.char_image and not args.char_prompt:
            parser.error("Either --char-image or --char-prompt must be provided")
        run_sprite_gen(
            char_image=args.char_image,
            char_prompt=args.char_prompt,
            out_dir=args.out_dir,
            **kwargs,
        )

    elif args.command in ("composite", "character-with-weapon"):
        run_composite_weapon(
            weapon_image=args.weapon,
            sprite_dir=args.sprite_dir,
            out_dir=args.out_dir,
            strength=args.strength,
            weapon_prompt=args.weapon_prompt,
            **kwargs,
        )

    elif args.command == "all":
        if not args.char_image and not args.char_prompt:
            parser.error("Either --char-image or --char-prompt must be provided")

        # Step 1: Generate weapon (or use existing)
        char_dir = os.path.join(script_dir, "dist", "character")
        weapon_dir = os.path.join(script_dir, "dist", "weapon")

        if args.weapon:
            weapon_path = args.weapon
            print(f"🟢 STEP 1/3: Using existing weapon sprite: {weapon_path}")
        else:
            print("=" * 60)
            print("🟢 STEP 1/3: Generate weapon sprite")
            print("=" * 60)
            weapon_path = os.path.join(script_dir, "dist", "weapon.png")
            run_single_weapon(
                prompt=args.weapon_prompt,
                output=weapon_path,
                seed=args.seed_base,
                steps=args.steps,
                width=args.width,
                height=args.height,
            )

        # Step 2: Generate character sprites
        print("\n" + "=" * 60)
        print("🟢 STEP 2/3: Generate character sprites")
        print("=" * 60)
        run_sprite_gen(
            char_image=args.char_image,
            char_prompt=args.char_prompt,
            out_dir=char_dir,
            **kwargs,
        )

        # Step 3: Composite weapon onto sprites
        print("\n" + "=" * 60)
        print("🟢 STEP 3/3: Composite weapon onto sprites")
        print("=" * 60)
        run_composite_weapon(
            weapon_image=weapon_path,
            sprite_dir=char_dir,
            out_dir=weapon_dir,
            weapon_prompt=args.weapon_prompt,
            **kwargs,
        )

        print("\n" + "=" * 60)
        print("✅ Pipeline complete!")
        print(f"   Weapon:       {weapon_path}")
        print(f"   Characters:   {char_dir}/")
        print(f"   Weaponized:   {weapon_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
