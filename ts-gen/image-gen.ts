#!/usr/bin/env bun

// Image generation utilities using mflux-generate-flux2.

import { $ } from "bun";

/** Get the attributes of a class, excluding methods and static members. */
type ClassAttrs<T> = { [K in keyof T as T[K] extends Function ? never : K]: T[K] };

export const theme = {
  app: "Cel-shaded manga, thick black outlines, extreme high-contrast inky base, vibrant neon accents, visible screentone textures",
  humans: "Crazy Demon Slayer style hair and boots",
} as const;

export type ImageGenProps = ConstructorParameters<typeof ImageGen>[0];

export class ImageGen {
  static transparencyColor = "#00FF00"; // Green screen background color

  height = 512;
  inputImages: string[] = [];
  prompt = "";
  seed = 42;
  steps = 4;
  width = 512;

  constructor(props: Partial<ClassAttrs<ImageGen>>) {
    Object.assign(this, props);
  }

  clone(): ImageGen {
    return new ImageGen({ ...this });
  }

  async gen(outPath: string) {
    console.log(`Generating image with params: `, { ...this, outPath });

    if (!this.prompt) {
      throw new Error("Prompt is required");
    }

    await $`
      mkdir -p $(dirname ${outPath})
      rm -rf ${outPath}
    `;

    if (this.inputImages.length > 0) {
      await $`
        mflux-generate-flux2-edit \
          --model flux2-klein-9b \
          --image-paths ${this.inputImages.join(' ')} \
          --prompt "${this.prompt}" \
          --steps ${this.steps} \
          --seed ${this.seed} \
          --width ${this.width} \
          --height ${this.height} \
          --output ${outPath}
      `;
    } else {
      await $`
        mflux-generate-flux2 \
          --model flux2-klein-9b \
          --prompt "${this.prompt}" \
          --steps ${this.steps} \
          --seed ${this.seed} \
          --width ${this.width} \
          --height ${this.height} \
          --output ${outPath}
      `;
    }

    console.log(`Saved generated image to ${outPath}`);

    // Save image as input for next step(s)
    this.inputImages = [outPath]
  }
}

export type CharacterDetails = {
  appearance: string,
}

export class CharacterGen extends ImageGen {
  constructor(deets: CharacterDetails & Omit<ImageGenProps, 'prompt'>) {
    super({
      ...deets,
      prompt: [
        "Make 2d pixel art sprite of a character viewed from front",
        "facing and looking at front",
        `background: solid ${ImageGen.transparencyColor} greenscreen`,
        theme.app,
        theme.humans,
        deets.appearance
      ].join('; '),
    });
  }

  async genRight(outPath: string) {
    if (!this.inputImages.length) {
      throw new Error("No input image provided");
    }
    const c = this.clone();
    c.prompt = [
      "Make the character side profile facing the right side of the screen",
    ].join('; ');
    await c.gen(outPath);
  }

  async genTopDown(outPath: string) {
    if (!this.inputImages.length) {
      throw new Error("No input image provided");
    }
    const c = this.clone();
    c.prompt = [
      "Top-down 2D retro sprite, overhead run-and-gun style, camera straight down",
      "Camera positioned high above looking straight down on the character like Grand Theft Auto GTA 2 style",
      "The head is big so all you can see is the top of the head"
    ].join('; ');
    await c.gen(outPath);
  }
}

export type RoomDetails = {
  appearance: string,
}

export class RoomGen extends ImageGen {
  constructor(deets: RoomDetails & Omit<ImageGenProps, 'prompt' | 'height' | 'width'>) {
    super({
      ...deets,
      height: 1024,
      prompt: [
        "Generate a complete room filling the entire canvas edge-to-edge with no borders, no background, no empty space",
        "Entire room fills frame edge-to-edge like a flat floor plan, thin walls form a solid border around all four sides",
        `Details: ${deets.appearance}`,
        "Pure 2D bird's-eye floor plan view, absolutely flat, zero perspective, zero vanishing points, zero depth, no 3D rendering, no isometric, no oblique angle",
        "Flat 2D pixel art, no shading, no gradients, no lighting, no shadows, crisp hard edges, like a SNES RPG map",
        "2D RPG room of a rectangular room with walls",
        "Camera pointed straight down at a perfect 90-degree angle, pure orthographic top-down",
      ].join('; '),
      width: 1024,
    });
  }
}
