# Image Generation Experiments 

Tips:

- img2img != txt2img with conditioning image(s). img2img has the ai start with the image and evolve. txt2img with conditioning image(s) is txt2img but AI trains on the images to copy the essence of the images -- way more capable for making major edits.
- model is optimized for 1024x1024
- Don't do dims less than 512x512
- Imagegens are known to have trouble with transparency - so may be better to have it do a solid background color and then remove it with ImageMagick.
- Imagegens are very inconsistent with making grids of sprites.

## mflux flux2-klein:9b

Seems the best in flexibility, capability, and performance at the moment. It can do most edit needs with the built-in CLI, and could easily be extended with Python too.

Note: The custom CLI python scripts mflux/flux.py does not currently get good results vs the official -- perhaps bc I was using img2img instead of txt2img with conditioning image(s).

### Install
You can run using uvx, or install to path using `uv tool install`. Note: will not download model(s) until first run
```bash
uv tool install mflux
```

### Usage TS-GEN

[./ts-gen/image-gen.ts](./ts-gen/image-gen.ts) is a Typescript interfaces to the mflux CLI for creating game media.

From [./ts-gen/alchemist-sprite-gen.ts](./ts-gen/alchemist-sprite-gen.ts),

```typescript
// Creates a sprite of a single character, a manga style alchemist

import { CharacterGen, type CharacterDetails } from "./image-gen.ts";

const alchemistDetails: CharacterDetails = {
  appearance: [
    "leather chest apron and a bandolier of glass flasks (amber, emerald, violet liquids)",
    "Soot-covered hands are visible on the periphery",
  ].join("; ")
}

const alchemistGen = new CharacterGen(alchemistDetails);
await alchemistGen.gen("dist/alchemist.png");
await alchemistGen.genRight("dist/alchemist-right.png");
await alchemistGen.genTopDown("dist/alchemist-top-down.png");
```

From [./ts-gen/room-gen.ts](./ts-gen/room-gen.ts),

```typescript
// Creates a 2D RPG room.

import { RoomGen, type RoomDetails } from "./image-gen.ts";

const roomDetails: RoomDetails = {
  appearance: [
    "haunted, gloomy, wood floor texture, wallpaper walls, window holes cut into walls seen from above",
    "2 door openings in walls seen from above, furniture seen from directly above as flat icons, purple hues",
  ].join("; ")
}

const roomGen = new RoomGen(roomDetails);
await roomGen.gen("dist/room.png");
```

### Usage CLI

Tips:
- use `uvx ...` if not installed to path
- big gen models will download model(s) on first run (15-30GB)

Ex. from https://github.com/filipstrand/mflux/blob/main/src/mflux/models/flux2/README.md

#### txt2img

```bash
mflux-generate-flux2 \
        --model flux2-klein-9b \
        --prompt "Make 2d pixel art sprite of a girl viewed from front; Has crazy hair; facing and looking at front" \
        --steps 4 \
        --seed 42 \
        --output dist/girl.png
```

#### img2img

img2img has the ai start from an image instead of random noise. It allows for making pixel-perfect edits, but doesn't seem to try as hard at understanding what the input image is and making edits based on that.

Here is an example of img2img to add glasses to a sprite character. Maybe because it is adding a new object to the image, I have to lower the image strength (alignment) to the point that the char doesn't look like the same char anymore.

```bash
mflux-generate-flux2 \
    --model flux2-klein-9b \
    --image-path refs/girl-warrior.png \
    --prompt "Make the girl wear glasses" \
    --image-strength 0.6 \
    --steps 4 \
    --seed 42 \
    --output dist/sprite-girl-glasses.png
```

### Mixing and guiding images (img2img+guiding)

This may differ slightly from model to model, but for Flux.2 there is an alternate image editing flow which seems to take inputs for inspiration rather than starting with them, which seems to result much better structural changes.

```bash
mflux-generate-flux2-edit \
    --model flux2-klein-9b \
    --image-paths refs/girl-warrior.png \
    --prompt "Make the girl wear glasses" \
    --steps 4 \
    --seed 42 \
    --output dist/sprite-girl-glasses-mixing.png

mflux-generate-flux2-edit \
    --model flux2-klein-9b \
    --image-paths refs/girl-warrior.png \
    --prompt "Top-down 2D retro sprite, overhead run-and-gun style, camera straight down; Camera positioned high above looking straight down on the character like Grand Theft Auto GTA 2 style; The hair is big so all you can see is hair" \
    --steps 4 \
    --seed 42 \
    --output dist/girl-top-down.png

mflux-generate-flux2-edit \
    --model flux2-klein-9b \
    --image-paths refs/front.png refs/blaster3.png \
    --prompt "Make the girl carrying the blaster horizontally with right hand at rear and left hand at front; fighting expression on face with closed mouth" \
    --steps 4 \
    --seed 42 \
    --output dist/girl-front-with-gun.png
    
mflux-generate-flux2-edit \
    --model flux2-klein-9b \
    --image-paths refs/girl.jpg refs/glasses.png \
    --prompt "Make the woman wear the eyeglasses (regular glasses, not sunglasses)" \
    --steps 4 \
    --seed 42 \
    --output dist/girl-wearing-glasses.png
```


## Ollama's flux2-klein:9b

Is MLX so Super performant, but limited. Does not include huggingface .safetensors files for the model, so we can't use native python features to access it, only the ollama CLI/api which is very limited in functionality and does not allow for things like Lora,image input. It has an image input feature, but it doesn't seem to work.

Example:

```bash
ollama run x/flux2-klein:9b --width=300 --height=300 "Generate a complete character sprite sheet view. Subject: Single complete figure of a alchemist. Details: Crazy Demon Slayer style hair and boots. leather chest apron and a bandolier of glass flasks (amber, emerald, violet liquids). Soot-covered hands are visible on the periphery.. Perspective: Absolute extreme vertical 90-degree top-down aerial view including the entire subject with extensive padding. Viewer: The viewer is looking directly at the very top of the subject. Composition: Minimalist composition for icon use. Zero depth of field. Style: Cel-shaded manga, thick black outlines, extreme high-contrast inky base, vibrant neon accents, visible screentone textures. Background: Flat solid #00FF00 green screen; no frames, no borders, no edge effects"
mv *.png hunter.png
```



## flux2-klein:4b

Works well for normal image gen and basic img2img. Stopped experimenting bc could not get it to reliably add a weapon to the character. But now I'm thinking about using Lora so may be considering going back to it if mflux doesn't work.

## flux1-dev:32b

Experimented a bit but stopped because very slow and was not getting good results. I may be able to get better results with tuning and Lora, but I'm switching focus to Flux2 for now.


## Comfy research

### Set 1:

https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors
https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors



### Set 2:

https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors
https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors
https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors


### Set 3:

https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors

## Set 4 (MLX Optimized for Apple Silicon):

Note: This model has built-in encoder/decoder so you would only need one.

https://huggingface.co/mlx-community/FLUX.2-klein-9B/resolve/main/flux_klein_9b_4bit.safetensors

### Alternative MLX Options:
- **FLUX.1 lite 8B Q8** (smaller, faster): https://huggingface.co/mlx-community/Flux-1.lite-8B-MLX-Q8/resolve/main/flux_1_lite_8b_q8_0.safetensors
- **FLUX.1 lite 8B Q4** (smallest, fastest): https://huggingface.co/mlx-community/Flux-1.lite-8B-MLX-Q4/resolve/main/flux_1_lite_8b_q4_0.safetensors

