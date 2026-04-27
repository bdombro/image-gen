#!/usr/bin/env bun

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
