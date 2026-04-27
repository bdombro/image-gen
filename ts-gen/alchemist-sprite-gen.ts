#!/usr/bin/env bun

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
